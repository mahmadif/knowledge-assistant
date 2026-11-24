# Incident Response Runbook (fictional)

## Severity levels
Severity 1 means a full outage of a customer-facing service. Severity 2 means a partial outage or a serious degradation. Severity 3 covers minor issues with a workaround.

## First response
The on-call engineer acknowledges the alert within 10 minutes. For Severity 1 incidents the on-call engineer opens an incident channel and appoints an incident commander. The incident commander coordinates the response and posts a status update every 30 minutes.

## Communication
Customer communication is handled by the support lead, never by the engineers directly. The status page is updated within 15 minutes of a confirmed Severity 1 incident.

## Postmortem
A blameless postmortem is written within five business days after every Severity 1 or Severity 2 incident. The postmortem lists the timeline, the root cause, and concrete follow-up actions with owners.
