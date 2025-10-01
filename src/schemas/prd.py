from typing import List
from pydantic import BaseModel


class Persona(BaseModel):
    name: str
    details: str


class Condition(BaseModel):
    label: str
    explanation: str


class FlowStep(BaseModel):
    order: int
    detail: str


class ExceptionCase(BaseModel):
    scenario: str
    handling: str


class Story(BaseModel):
    order: int
    title: str
    subject_role: str
    action_goal: str
    business_value: str
    personas: List[Persona]
    pre_conditions: List[Condition]
    done_when: List[FlowStep]
    uses_flow_diagram: bool = False
    exception_handling: List[ExceptionCase]
    acceptance_criteria: List[str]
    definition_of_done: List[str]


class PRDTemplateSchema(BaseModel):
    feature: str
    introduction: str
    user_stories: List[Story]
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    assumptions: List[str]
    dependencies: List[str]
    risks_and_mitigations: List[str]
    timeline: str
    stakeholders: List[str]
    metrics: List[str]
