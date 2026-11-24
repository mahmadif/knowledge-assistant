# Engineering Standards (fictional)

## Code review
Every change needs one approving review before it is merged. Security-sensitive changes, such as authentication or payment code, need two approving reviews, one of them from a member of the security team.

## Quality gates
The continuous integration pipeline must be green before merging. The target test coverage for new code is 80 percent.

## Branches and commits
Branch names follow the pattern type/ticket-id-short-description. Commit messages are written in the imperative mood.

## Architecture decisions
Significant architecture decisions are documented as short architecture decision records in the repository.
