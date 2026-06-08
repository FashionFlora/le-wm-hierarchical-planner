"""JEPA variant whose planner state is hierarchical Z15/Z5/Z1."""

import torch
import torch.nn.functional as F
from einops import rearrange

from jepa import JEPA


class PlanningJEPA(JEPA):
    """LeWM planner where Z15/Z5/Z1 are the world state.

    `emb` is kept as a compatibility readout for the existing LeWM API, but the
    planner predictor and planner loss operate directly on the structured state.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        state_encoder=None,
        state_refiner=None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.state_encoder = state_encoder or state_refiner

    def encode(self, info):
        """Encode pixels into the hierarchical state plus an `emb` readout."""

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)

        tokens = output.last_hidden_state
        if self.state_encoder is None:
            cls_emb = tokens[:, 0]
            emb = self.projector(cls_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        else:
            state = self.state_encoder(tokens[:, 1:])
            info["emb"] = rearrange(state["emb"], "(b t) d -> b t d", b=b)
            for key in ("z15", "z5", "z1", "fused_z15"):
                if key in state:
                    info[key] = rearrange(state[key], "(b t) n d -> b t n d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict_state(self, state_history, act_emb):
        """Predict next Z15/Z5/Z1 states and attach an `emb` readout."""

        pred_state = self.predictor(state_history, act_emb)
        if self.state_encoder is not None:
            b, t = pred_state["z15"].shape[:2]
            flat_state = {
                key: rearrange(value, "b t n d -> (b t) n d")
                for key, value in pred_state.items()
            }
            flat_state["emb"] = self.state_encoder.readout_state(flat_state)
            pred_state["emb"] = rearrange(flat_state["emb"], "(b t) d -> b t d", b=b, t=t)
        return pred_state

    def predict(self, emb, act_emb):
        """Compatibility path for vanilla embedding predictors."""

        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Roll out the hierarchical world state autoregressively.

        This is the missing part of the full planner idea: the future is a
        sequence of Z15/Z5/Z1 states. `predicted_emb` is only the readout used by
        the existing cost API.
        """

        assert "pixels" in info, "pixels not in info_dict"
        hist = info["pixels"].size(2)
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [hist, horizon - hist], dim=2)
        info["action"] = act_0
        n_steps = horizon - hist

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)

        state = {}
        for key in ("z15", "z5", "z1"):
            state[key] = init[key].unsqueeze(1).expand(batch_size, num_samples, *init[key].shape[1:])
            state[key] = rearrange(state[key], "b s ... -> (b s) ...").clone()

        emb = init["emb"].unsqueeze(1).expand(batch_size, num_samples, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        hs = history_size
        for step in range(n_steps):
            act_emb = self.action_encoder(act)
            state_trunc = {key: value[:, -hs:] for key, value in state.items()}
            act_trunc = act_emb[:, -hs:]
            pred_state = self.predict_state(state_trunc, act_trunc)

            for key in ("z15", "z5", "z1"):
                state[key] = torch.cat([state[key], pred_state[key][:, -1:]], dim=1)
            emb = torch.cat([emb, pred_state["emb"][:, -1:]], dim=1)

            next_act = act_future[:, step : step + 1]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        state_trunc = {key: value[:, -hs:] for key, value in state.items()}
        pred_state = self.predict_state(state_trunc, act_emb[:, -hs:])
        for key in ("z15", "z5", "z1"):
            state[key] = torch.cat([state[key], pred_state[key][:, -1:]], dim=1)
        emb = torch.cat([emb, pred_state["emb"][:, -1:]], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=batch_size, s=num_samples)
        for key in ("z15", "z5", "z1"):
            info[f"predicted_{key}"] = rearrange(
                state[key], "(b s) ... -> b s ...", b=batch_size, s=num_samples
            )
        return info

    def criterion(self, info_dict: dict):
        """Planning cost on the hierarchical world state when available."""

        if all(k in info_dict for k in ("predicted_z15", "predicted_z5", "predicted_z1", "goal_z15", "goal_z5", "goal_z1")):
            total = None
            for key, weight in (("z15", 1.0), ("z5", 0.5), ("z1", 0.25)):
                pred = info_dict[f"predicted_{key}"][..., -1:, :, :]
                goal = info_dict[f"goal_{key}"][..., -1:, :, :].unsqueeze(1)
                goal = goal.expand_as(pred)
                cost = F.mse_loss(pred, goal.detach(), reduction="none").sum(
                    dim=tuple(range(2, pred.ndim))
                )
                total = weight * cost if total is None else total + weight * cost
            return total

        return super().criterion(info_dict)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute MPC cost by rolling out Z15/Z5/Z1, not only the emb readout."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in list(info_dict.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        for key in ("z15", "z5", "z1"):
            info_dict[f"goal_{key}"] = goal[key]

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)

