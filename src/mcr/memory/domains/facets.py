"""Unified schema-v2 facet models shared by all six memory domains."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mcr.memory.common import DomainModel, NonEmptyStr
from mcr.memory.domains.base import DescriptionBase


def _optional():
    return Field(default=None, exclude_if=lambda value: value is None)


class CoreAttributes(DomainModel):
    subject: NonEmptyStr | None = _optional()
    predicate: NonEmptyStr | None = _optional()
    object: NonEmptyStr | None = _optional()


class EntityTypeAttributes(DomainModel):
    entity: NonEmptyStr | None = _optional()
    type: NonEmptyStr | None = _optional()


class AttributeAttributes(DomainModel):
    entity: NonEmptyStr | None = _optional()
    name: NonEmptyStr | None = _optional()
    value: str | int | float | bool | None = _optional()
    unit: NonEmptyStr | None = _optional()


class QuantityAttributes(DomainModel):
    entity: NonEmptyStr | None = _optional()
    amount: int | float | None = _optional()
    unit: NonEmptyStr | None = _optional()


class CostAttributes(DomainModel):
    item: NonEmptyStr | None = _optional()
    amount: int | float | None = _optional()
    currency: NonEmptyStr | None = _optional()


class PaymentMethodAttributes(DomainModel):
    item: NonEmptyStr | None = _optional()
    method: NonEmptyStr | None = _optional()


class TimeAttributes(DomainModel):
    event: NonEmptyStr | None = _optional()
    value: NonEmptyStr | None = _optional()
    normalized_value: NonEmptyStr | None = _optional()
    precision: Literal["exact", "day", "week", "month", "year", "relative", "unknown"] | None = _optional()


class LocationAttributes(DomainModel):
    event: NonEmptyStr | None = _optional()
    place: NonEmptyStr | None = _optional()
    city: NonEmptyStr | None = _optional()
    country: NonEmptyStr | None = _optional()


class ParticipantAttributes(DomainModel):
    event: NonEmptyStr | None = _optional()
    person: NonEmptyStr | None = _optional()
    role: NonEmptyStr | None = _optional()


class RelationshipAttributes(DomainModel):
    person: NonEmptyStr | None = _optional()
    relationship_type: NonEmptyStr | None = _optional()
    target: NonEmptyStr | None = _optional()


class EmotionAttributes(DomainModel):
    event: NonEmptyStr | None = _optional()
    emotion: NonEmptyStr | None = _optional()
    intensity: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)


class EvaluationAttributes(DomainModel):
    target: NonEmptyStr | None = _optional()
    sentiment: Literal["positive", "neutral", "negative", "mixed"] | None = _optional()
    strength: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)


class PreferenceAttributes(DomainModel):
    target: NonEmptyStr | None = _optional()
    polarity: float | None = Field(default=None, ge=-1, le=1, exclude_if=lambda value: value is None)
    strength: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)


class PurposeAttributes(DomainModel):
    action: NonEmptyStr | None = _optional()
    purpose: NonEmptyStr | None = _optional()


class StatusAttributes(DomainModel):
    subject: NonEmptyStr | None = _optional()
    status: NonEmptyStr | None = _optional()


class OutcomeAttributes(DomainModel):
    action: NonEmptyStr | None = _optional()
    outcome: NonEmptyStr | None = _optional()


class FrequencyAttributes(DomainModel):
    action: NonEmptyStr | None = _optional()
    frequency: NonEmptyStr | None = _optional()


class DurationAttributes(DomainModel):
    event: NonEmptyStr | None = _optional()
    value: int | float | None = _optional()
    unit: NonEmptyStr | None = _optional()


class ScoreAttributes(DomainModel):
    framework: NonEmptyStr | None = _optional()
    trait: NonEmptyStr | None = _optional()
    value: int | float | None = _optional()
    scale: NonEmptyStr | None = _optional()


class CoreFacet(DescriptionBase):
    facet_type: Literal["core"]
    attributes: CoreAttributes


class EntityTypeFacet(DescriptionBase):
    facet_type: Literal["entity_type"]
    attributes: EntityTypeAttributes


class AttributeFacet(DescriptionBase):
    facet_type: Literal["attribute"]
    attributes: AttributeAttributes


class QuantityFacet(DescriptionBase):
    facet_type: Literal["quantity"]
    attributes: QuantityAttributes


class CostFacet(DescriptionBase):
    facet_type: Literal["cost"]
    attributes: CostAttributes


class PaymentMethodFacet(DescriptionBase):
    facet_type: Literal["payment_method"]
    attributes: PaymentMethodAttributes


class TimeFacet(DescriptionBase):
    facet_type: Literal["time"]
    attributes: TimeAttributes


class LocationFacet(DescriptionBase):
    facet_type: Literal["location"]
    attributes: LocationAttributes


class ParticipantFacet(DescriptionBase):
    facet_type: Literal["participant"]
    attributes: ParticipantAttributes


class RelationshipFacet(DescriptionBase):
    facet_type: Literal["relationship"]
    attributes: RelationshipAttributes


class EmotionFacet(DescriptionBase):
    facet_type: Literal["emotion"]
    attributes: EmotionAttributes


class EvaluationFacet(DescriptionBase):
    facet_type: Literal["evaluation"]
    attributes: EvaluationAttributes


class PreferenceFacet(DescriptionBase):
    facet_type: Literal["preference"]
    attributes: PreferenceAttributes


class PurposeFacet(DescriptionBase):
    facet_type: Literal["purpose"]
    attributes: PurposeAttributes


class StatusFacet(DescriptionBase):
    facet_type: Literal["status"]
    attributes: StatusAttributes


class OutcomeFacet(DescriptionBase):
    facet_type: Literal["outcome"]
    attributes: OutcomeAttributes


class FrequencyFacet(DescriptionBase):
    facet_type: Literal["frequency"]
    attributes: FrequencyAttributes


class DurationFacet(DescriptionBase):
    facet_type: Literal["duration"]
    attributes: DurationAttributes


class ScoreFacet(DescriptionBase):
    facet_type: Literal["score"]
    attributes: ScoreAttributes


DescriptionFacet = Annotated[
    CoreFacet | EntityTypeFacet | AttributeFacet | QuantityFacet | CostFacet
    | PaymentMethodFacet | TimeFacet | LocationFacet | ParticipantFacet
    | RelationshipFacet | EmotionFacet | EvaluationFacet | PreferenceFacet
    | PurposeFacet | StatusFacet | OutcomeFacet | FrequencyFacet | DurationFacet
    | ScoreFacet,
    Field(discriminator="facet_type"),
]
