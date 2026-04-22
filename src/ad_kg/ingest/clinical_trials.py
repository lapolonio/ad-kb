"""ClinicalTrials.gov ingest using API v2."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ad_kg.models import Trial

logger = logging.getLogger(__name__)

_CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
_PAGE_SIZE = 100


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get_page(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a single page from the ClinicalTrials v2 API."""
    resp = requests.get(_CT_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_study(study: dict[str, Any]) -> Trial | None:
    """Parse a ClinicalTrials v2 study dict into a Trial object."""
    try:
        proto = study.get("protocolSection", {})
        id_module = proto.get("identificationModule", {})
        status_module = proto.get("statusModule", {})
        design_module = proto.get("designModule", {})
        conditions_module = proto.get("conditionsModule", {})
        arms_module = proto.get("armsInterventionsModule", {})
        desc_module = proto.get("descriptionModule", {})

        nct_id = id_module.get("nctId", "")
        if not nct_id:
            return None

        title = id_module.get("briefTitle", "") or id_module.get("officialTitle", "")
        status = status_module.get("overallStatus", "")
        phases = design_module.get("phases", [])
        phase = ", ".join(phases) if phases else "N/A"
        conditions = conditions_module.get("conditions", [])

        interventions: list[str] = []
        for iv in arms_module.get("interventions", []):
            name = iv.get("name", "")
            if name:
                interventions.append(name)

        summary = desc_module.get("briefSummary", "") or ""

        return Trial(
            nct_id=nct_id,
            title=title,
            status=status,
            phase=phase,
            conditions=conditions,
            interventions=interventions,
            summary=summary,
        )
    except Exception as exc:
        logger.warning("Failed to parse study: %s", exc)
        return None


def fetch_clinical_trials(
    condition: str = "Alzheimer's disease",
    interventions: list[str] | None = None,
    limit: int | None = None,
) -> list[Trial]:
    """Fetch clinical trials from ClinicalTrials.gov API v2.

    Args:
        condition: Disease/condition to filter by.
        interventions: Optional list of intervention names to filter.
        limit: Maximum number of trials to return (None = no limit).

    Returns:
        Deduplicated list of Trial objects.
    """
    logger.info("Fetching clinical trials for condition=%r", condition)

    params: dict[str, Any] = {
        "query.cond": condition,
        "pageSize": min(_PAGE_SIZE, limit) if limit else _PAGE_SIZE,
        "format": "json",
    }
    if interventions:
        params["query.intr"] = "|".join(interventions)

    seen_nct: set[str] = set()
    trials: list[Trial] = []
    next_token: str | None = None

    while True:
        if next_token:
            params["pageToken"] = next_token
        elif "pageToken" in params:
            del params["pageToken"]

        try:
            data = _get_page(params)
        except Exception as exc:
            logger.error("ClinicalTrials API failed: %s", exc)
            break

        studies = data.get("studies", [])
        for study in studies:
            trial = _parse_study(study)
            if trial is None:
                continue
            if trial.nct_id in seen_nct:
                continue
            seen_nct.add(trial.nct_id)
            trials.append(trial)
            if limit and len(trials) >= limit:
                logger.info("Reached limit of %d trials", limit)
                return trials

        next_token = data.get("nextPageToken")
        if not next_token:
            break

    logger.info("Fetched %d unique trials", len(trials))
    return trials
