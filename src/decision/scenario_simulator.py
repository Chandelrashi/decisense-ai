from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import copy


@dataclass
class Scenario:
    name: str
    changes: Dict[str, Any]
    description: str


def default_scenarios() -> List[Scenario]:
    """
    Predefined scenario set (MVP).
    These align with your questionnaire levers and are easy to explain to reviewers.
    """
    return [
        Scenario(
            name="Baseline (no changes)",
            changes={"planned_marketing_change": "0", "planned_retention_invest": "0", "hiring_plan": "Replace only"},
            description="Current plan with no additional interventions."
        ),
        Scenario(
            name="Increase marketing (+10%)",
            changes={"planned_marketing_change": "+10"},
            description="Boost acquisition demand to drive short-term growth."
        ),
        Scenario(
            name="Increase marketing (+20%)",
            changes={"planned_marketing_change": "+20"},
            description="Aggressive growth push via marketing spend."
        ),
        Scenario(
            name="Retention investment (+5%)",
            changes={"planned_retention_invest": "+5"},
            description="Improve retention and reduce attrition risk."
        ),
        Scenario(
            name="Retention investment (+10%)",
            changes={"planned_retention_invest": "+10"},
            description="Stronger retention push to stabilise workforce during growth."
        ),
        Scenario(
            name="Net new hires",
            changes={"hiring_plan": "Net new hires"},
            description="Increase capacity to reduce strain and support growth."
        ),
        Scenario(
            name="Hiring freeze",
            changes={"hiring_plan": "Freeze"},
            description="Reduce cost but may increase strain and slow growth."
        ),
        Scenario(
            name="Balanced: marketing +10% & retention +5%",
            changes={"planned_marketing_change": "+10", "planned_retention_invest": "+5"},
            description="Grow while managing workforce risk."
        ),
    ]


def apply_changes(base_input: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    x = copy.deepcopy(base_input)
    for k, v in changes.items():
        x[k] = v
    return x


def generate_scenario_inputs(base_input: Dict[str, Any], scenarios: List[Scenario]) -> List[Dict[str, Any]]:
    """
    Returns list of dicts where each dict is one scenario input row (same schema as model features).
    """
    out = []
    for sc in scenarios:
        sc_input = apply_changes(base_input, sc.changes)
        sc_input["_scenario_name"] = sc.name
        sc_input["_scenario_description"] = sc.description
        out.append(sc_input)
    return out
