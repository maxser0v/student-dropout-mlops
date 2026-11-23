"""Small immutable value types at the platform worker boundary."""

from __future__ import annotations

from dataclasses import dataclass


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class AssignmentScope:
    institution_id: str
    course_id: str
    assignment_id: str

    def __post_init__(self) -> None:
        for name in ("institution_id", "course_id", "assignment_id"):
            _required(getattr(self, name), name)


@dataclass(frozen=True)
class ReportVersion:
    scope: AssignmentScope
    report_id: str
    version_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AssignmentScope):
            raise ValueError("scope must be an AssignmentScope")
        for name in ("report_id", "version_id"):
            _required(getattr(self, name), name)
        _required(self.text, "text")

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        """Deterministic scoped identity; versions of one report share report_id."""
        return (self.scope.institution_id, self.scope.course_id,
                self.scope.assignment_id, self.report_id, self.version_id)
