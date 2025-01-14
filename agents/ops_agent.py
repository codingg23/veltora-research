"""
ops_agent.py

LLM-based operations planning agent for infrastructure events.

Switched from RL to LLM-based planning for this one because:
1. The action space is too large and too structured for RL to be practical
2. Operators need to *understand* the reasoning, not just get a recommendation
3. LLMs handle novel situations (new failure modes) much better than a trained policy

The agent receives an infrastructure event, reasons about it using
the digital twin simulation as a tool, and produces a prioritised
action plan for the operations team.

This is early. The main thing missing is a feedback loop — right now
the agent makes recommendations but doesn't learn whether they worked.
That's the next thing to build.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional
import anthropic

logger = logging.getLogger(__name__)

# The system prompt for the ops agent is more detailed than the query agent
# because it needs to reason about complex infrastructure relationships.
OPS_SYSTEM_PROMPT = """You are an infrastructure operations intelligence system for a data centre.

Your job is to:
1. Analyse infrastructure events and anomalies
2. Determine root cause or likely cause based on available telemetry
3. Generate a prioritised action plan for the operations team
4. Use simulation tools to validate proposed actions before recommending them
5. Escalate clearly when the situation is outside normal bounds

Principles:
- Be decisive. Operators need a clear recommendation, not a list of possibilities.
- Show your reasoning. Explain *why* you're recommending each action.
- Quantify impact where possible. "CRAC at 94% capacity" is more useful than "cooling is stressed".
- Flag if human judgment is required. Some decisions need an experienced engineer.
- Be conservative in safety-critical situations. When in doubt, recommend the safer action.

You have access to tools for querying real telemetry data and running simulations.
Always use these tools before making recommendations — do not reason from assumptions alone.
"""

# Event severity levels
SEVERITY_LEVELS = {
    "info": 0,
    "warning": 1,
    "alert": 2,
    "critical": 3,
    "emergency": 4,
}


@dataclass
class InfraEvent:
    event_id: str
    event_type: str    # e.g., "cooling_degradation", "power_anomaly", "thermal_threshold"
    severity: str
    source: str        # e.g., "CRAC-04", "PDU-R07", "R07-12"
    description: str
    telemetry: Optional[dict] = None   # raw telemetry context at time of event
    timestamp: Optional[str] = None


@dataclass
class ActionPlan:
    event_id: str
    root_cause_hypothesis: str
    confidence: str    # "high", "medium", "low"
    immediate_actions: list[dict]    # [{team, action, urgency, rationale}]
    monitoring_actions: list[dict]
    escalation_required: bool
    estimated_impact_if_unaddressed: str
    simulation_results: Optional[dict] = None


class OpsAgent:
    """
    Infrastructure operations planning agent.

    Takes an InfraEvent, reasons about it with available tools,
    produces an ActionPlan.
    """

    def __init__(self, data_path: str, sim_model_path: Optional[str] = None):
        self.client = anthropic.Anthropic()
        self.data_path = data_path
        self.sim_model_path = sim_model_path

        # tools available to the ops agent
        self.tools = self._build_tools()

    def _build_tools(self) -> list[dict]:
        return [
            {
                "name": "get_recent_telemetry",
                "description": "Get recent sensor readings for a specific component or zone. Use to understand current state before recommending actions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "component_id": {"type": "string", "description": "CRAC ID, rack ID, row ID, etc."},
                        "lookback_minutes": {"type": "integer", "default": 60},
                        "metrics": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["component_id"],
                },
            },
            {
                "name": "run_failure_simulation",
                "description": "Simulate the impact of a component failure using the digital twin. Use to assess blast radius before deciding escalation level.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "component_type": {"type": "string", "enum": ["crac", "pdu", "ups", "chiller"]},
                        "component_id": {"type": "string"},
                        "failure_mode": {"type": "string", "enum": ["full_failure", "partial_50pct", "degraded_25pct"]},
                    },
                    "required": ["component_type", "component_id", "failure_mode"],
                },
            },
            {
                "name": "get_maintenance_schedule",
                "description": "Check scheduled maintenance for a component. Helps determine if an anomaly might be related to planned work.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "component_id": {"type": "string"},
                        "lookahead_days": {"type": "integer", "default": 30},
                    },
                    "required": ["component_id"],
                },
            },
            {
                "name": "check_related_events",
                "description": "Check if similar events have occurred recently. Pattern recognition across events.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_type": {"type": "string"},
                        "zone": {"type": "string"},
                        "lookback_days": {"type": "integer", "default": 7},
                    },
                    "required": ["event_type"],
                },
            },
        ]

    def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """
        Execute a tool call. For the prototype, most of these return
        mock/placeholder data. The real implementations would query
        DCIM, the simulation model, CMDB, etc.
        """
        # NOTE: These are stub implementations for the prototype.
        # Real implementations are in tools.py and the sim model.
        if tool_name == "get_recent_telemetry":
            return self._mock_telemetry(tool_input)
        elif tool_name == "run_failure_simulation":
            return self._mock_simulation(tool_input)
        elif tool_name == "get_maintenance_schedule":
            return {"scheduled_maintenance": [], "note": "No scheduled maintenance found"}
        elif tool_name == "check_related_events":
            return {"related_events": [], "note": "No similar recent events"}
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _mock_telemetry(self, args: dict) -> dict:
        """Placeholder — replace with real DCIM query."""
        return {
            "component_id": args.get("component_id"),
            "readings": {
                "crac_load_pct": 94,
                "supply_temp_c": 17.2,
                "return_temp_c": 29.8,
                "airflow_m3_per_s": 2.7,
                "compressor_current_a": 42.1,
            },
            "trend": "load increasing over past 45 minutes",
            "note": "Mock data — replace with real DCIM integration",
        }

    def _mock_simulation(self, args: dict) -> dict:
        """Placeholder — replace with real digital twin."""
        return {
            "scenario": f"{args.get('component_type')} {args.get('component_id')} {args.get('failure_mode')}",
            "projected_impact": {
                "affected_racks": 24,
                "max_inlet_temp_c": 31.2,
                "time_to_ashrae_limit_minutes": 18,
                "cooling_headroom_remaining_pct": 12,
            },
            "note": "Mock simulation — replace with real digital twin integration",
        }

    def analyse_event(self, event: InfraEvent, verbose: bool = False) -> ActionPlan:
        """
        Analyse an infrastructure event and produce an action plan.
        """
        user_message = f"""
Infrastructure Event Received:

Event ID:    {event.event_id}
Type:        {event.event_type}
Severity:    {event.severity.upper()}
Source:      {event.source}
Time:        {event.timestamp or 'unknown'}
Description: {event.description}

Raw telemetry context:
{json.dumps(event.telemetry or {}, indent=2)}

Please analyse this event, use the available tools to gather more context,
run relevant simulations if appropriate, and produce a clear action plan for
the operations team. Structure your response as:

1. Root cause hypothesis (with confidence level)
2. Immediate actions (what needs to happen in the next 30 minutes)
3. Follow-up actions (what needs to happen in the next 24 hours)
4. Escalation decision (should a senior engineer be paged?)
5. Impact estimate if left unaddressed
"""
        messages = [{"role": "user", "content": user_message}]
        raw_response = ""

        for _ in range(5):  # max tool call rounds
            response = self.client.messages.create(
                model="claude-opus-4-6",  # using Opus for complex reasoning
                max_tokens=2048,
                system=OPS_SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
            )

            if verbose:
                logger.info(f"stop_reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                raw_response = " ".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                break

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # for now, return a structured plan with the raw response as the primary content
        # TODO: parse the response into the ActionPlan fields more precisely
        return ActionPlan(
            event_id=event.event_id,
            root_cause_hypothesis="See full analysis below",
            confidence="medium",
            immediate_actions=[{"action": raw_response, "team": "operations", "urgency": event.severity}],
            monitoring_actions=[],
            escalation_required=SEVERITY_LEVELS.get(event.severity, 0) >= 2,
            estimated_impact_if_unaddressed="See simulation results",
        )
