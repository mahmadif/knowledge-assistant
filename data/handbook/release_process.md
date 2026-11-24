# Release Process (fictional)

## Release candidates
A release candidate must pass all automated tests and a completed code review before it can be promoted. The candidate is then deployed to the staging environment, where it must run for 24 hours without critical errors.

## Approval
A release needs two approvals: one from the release manager and one from the QA lead. Without both approvals the release cannot be started.

## Release windows
Regular releases are not allowed on Fridays after 14:00. Exceptions need the approval of the engineering manager.

## Hotfixes
A hotfix may skip the 24 hour staging period, but it still needs both approvals and a code review. Hotfixes for production incidents follow the incident response runbook.
