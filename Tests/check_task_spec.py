#!/usr/bin/env python3

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "HSGeneral": "RunSection/General/HS/HSExecutionPlan.cpp",
    "SSGeneral": "RunSection/General/SS/SSExecutionPlan.cpp",
    "MultiSSGeneral": "RunSection/General/MultiSS/MultiSSExecutionPlan.cpp",
}
REQUIRED_CONSTRAINTS = {
    "HSGeneral": {
        "powdergrid=uniform requires powdersamplingpoints",
        "powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
        "stochasticstandarddeviation=true requires sampling=stochastic",
        "observables=transitionyields requires calculation=yields",
        "powderdomain conflicts with powderfullsphere",
    },
    "SSGeneral": {
        "powdergrid=uniform requires powdersamplingpoints",
        "powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
    },
    "MultiSSGeneral": {
        "powdergrid=uniform requires powdersamplingpoints",
        "powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
    },
}
VALUE_TYPES = {"bool", "int", "float", "string", "list", "vector"}


def fail(message):
    raise AssertionError(message)


def run(binary, *arguments):
    return subprocess.run(
        [binary, *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def plan_input_keys(task_name, relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    keys = set(
        re.findall(
            rf"(?:GetTaskProperty|GetTaskPropertyList|GetTaskPulseSequence)\s*\(\s*\w+\s*,\s*\"{task_name}\"\s*,\s*\"([^\"]+)\"",
            source,
        )
    )
    keys.update(
        re.findall(r"\bReadString\s*\(\s*\w+\s*,\s*\"([^\"]+)\"", source)
    )
    keys.update(
        re.findall(r"\breadTimelineWindow\s*\(\s*\"([^\"]+)\"", source)
    )
    return keys


def check_condition(condition):
    if condition is None:
        return
    if not isinstance(condition, dict):
        fail(f"Invalid condition shape: {condition!r}")
    if "key" in condition:
        operators = set(condition) - {"key"}
        if not isinstance(condition["key"], str) or len(operators) != 1:
            fail(f"Invalid comparison condition: {condition!r}")
        if not operators.issubset({"equals", "not_equals", "in", "present", "gt", "ge", "lt", "le"}):
            fail(f"Unsupported comparison operator: {operators}")
        return
    if len(condition) != 1:
        fail(f"Invalid condition shape: {condition!r}")
    operator, value = next(iter(condition.items()))
    if operator in {"all", "any"}:
        if not isinstance(value, list):
            fail(f"Condition {operator!r} must contain a list")
        for child in value:
            check_condition(child)
    elif operator == "not":
        check_condition(value)
    else:
        fail(f"Unsupported condition operator: {operator}")


def check_spec(binary):
    first = run(binary, "--dump-task-spec")
    second = run(binary, "--dump-task-spec")
    if first.returncode != 0:
        fail(f"All-task dump exited {first.returncode}: {first.stderr}")
    if first.stderr:
        fail(f"All-task dump wrote to stderr: {first.stderr}")
    if first.stdout != second.stdout:
        fail("Repeated all-task dumps differ byte-for-byte")

    spec = json.loads(first.stdout)
    if spec.get("schema_version") != 1:
        fail("schema_version must equal 1")
    if spec.get("molspin_version") != "2.3":
        fail("molspin_version must match the executable's current version")
    if set(spec.get("tasks", {})) != set(TASKS):
        fail("Task set does not match the supported general tasks")

    factory = (ROOT / "RunSection/RunSection_CreateTask.cpp").read_text(encoding="utf-8")
    for task_name, source_path in TASKS.items():
        task = spec["tasks"][task_name]
        plan_source = (ROOT / source_path).read_text(encoding="utf-8")
        if task.get("type") != task_name:
            fail(f"{task_name}: wrong canonical type")
        if f'IsTaskSpecAlias("{task_name}", _tasktype)' not in factory:
            fail(f"{task_name}: task creation does not consume declared aliases")
        if not task.get("aliases") or len(set(task["aliases"])) != len(task["aliases"]):
            fail(f"{task_name}: task aliases must be non-empty and unique")

        properties = task.get("properties")
        if not isinstance(properties, dict) or not properties:
            fail(f"{task_name}: properties must be a non-empty object")

        for key, prop in properties.items():
            if prop.get("name") != key:
                fail(f"{task_name}.{key}: name does not match its object key")
            if prop.get("value_type") not in VALUE_TYPES:
                fail(f"{task_name}.{key}: invalid value_type")
            if not isinstance(prop.get("aliases"), list):
                fail(f"{task_name}.{key}: aliases must be a list")
            if not isinstance(prop.get("choices"), list):
                fail(f"{task_name}.{key}: choices must be a list")
            if not isinstance(prop.get("required"), bool):
                fail(f"{task_name}.{key}: required must be boolean")
            if not isinstance(prop.get("description"), str) or "\n" in prop["description"]:
                fail(f"{task_name}.{key}: description must be one line")
            if len(set(prop["aliases"])) != len(prop["aliases"]):
                fail(f"{task_name}.{key}: aliases must be unique")

        source_keys = plan_input_keys(task_name, source_path)
        declared_keys = set(properties)
        if source_keys != declared_keys:
            missing = sorted(source_keys - declared_keys)
            stale = sorted(declared_keys - source_keys)
            fail(f"{task_name}: parser/spec declaration drift; missing={missing}, stale={stale}")

        constraints = task.get("constraints")
        if not isinstance(constraints, list):
            fail(f"{task_name}: constraints must be a list")
        messages = set()
        for constraint in constraints:
            if constraint.get("kind") not in {"requires", "conflicts", "scoped", "range"}:
                fail(f"{task_name}: invalid constraint kind")
            if not isinstance(constraint.get("keys"), list) or not constraint["keys"]:
                fail(f"{task_name}: constraint keys must be a non-empty list")
            if not set(constraint["keys"]).issubset(properties):
                fail(f"{task_name}: constraint refers to an unknown property")
            if not isinstance(constraint.get("message"), str):
                fail(f"{task_name}: constraint message must be a string")
            if constraint["when"] is None:
                fail(f"{task_name}: property-based constraints must be structured")
            check_condition(constraint["when"])
            if constraint["message"] not in plan_source:
                fail(f"{task_name}: declared constraint message is not present in the validator")
            messages.add(constraint["message"])
        if not REQUIRED_CONSTRAINTS[task_name].issubset(messages):
            fail(f"{task_name}: required existing validation messages are missing")

    hs_properties = spec["tasks"]["HSGeneral"]["properties"]
    if hs_properties["totaltime"]["value_type"] != "float" or hs_properties["totaltime"]["default"] != 10000.0:
        fail("HSGeneral.totaltime metadata differs from its execution plan")
    if hs_properties["krylovsize"]["value_type"] != "int" or hs_properties["krylovsize"]["default"] != 16:
        fail("HSGeneral.krylovsize metadata differs from its execution plan")
    if hs_properties["spinlist"]["value_type"] != "list":
        fail("HSGeneral.spinlist must be a list")
    if not {"direct", "stochastic"}.issubset(hs_properties["sampling"]["choices"]):
        fail("HSGeneral.sampling choices are incomplete")
    if not {"exponential", "rk4"}.issubset(
        spec["tasks"]["SSGeneral"]["properties"]["propagationmethod"]["choices"]
    ):
        fail("SSGeneral.propagationmethod choices are incomplete")
    if spec["tasks"]["SSGeneral"]["properties"]["propagationmethod"]["default"] != "exponential":
        fail("SSGeneral propagation default differs from its execution plan")
    if spec["tasks"]["MultiSSGeneral"]["properties"]["propagationmethod"]["default"] != "rk4":
        fail("MultiSSGeneral propagation default differs from its execution plan")
    if hs_properties["powdergrid"]["default"] != "uniform":
        fail("HSGeneral powder grid default differs from its execution plan")

    for task_name in TASKS:
        filtered = run(binary, f"--dump-task-spec={task_name}")
        if filtered.returncode != 0 or filtered.stderr:
            fail(f"{task_name} filtered dump failed: {filtered.stderr}")
        if set(json.loads(filtered.stdout)["tasks"]) != {task_name}:
            fail(f"{task_name} filtered dump contains unexpected tasks")

    invalid = run(binary, "--dump-task-spec=NoSuchTask")
    if invalid.returncode == 0 or invalid.stdout:
        fail("Unknown task filter must fail without writing JSON to stdout")
    extra_arguments = run(binary, "--dump-task-spec", "HSGeneral")
    if extra_arguments.returncode == 0 or extra_arguments.stdout:
        fail("Task-spec option must reject additional arguments")


if __name__ == "__main__":
    executable = sys.argv[1] if len(sys.argv) > 1 else "./molspin"
    check_spec(str(Path(executable).resolve()))
    print("Task specification checks passed.")
