"""Prompt templates for LLM merge conflict resolution."""

import os
from conflict.parser import ConflictBlock

# Language detection by file extension
_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".sh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "",
}


def _detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return _EXT_TO_LANG.get(ext, "")


# ---------- Format A: whole_file ----------

WHOLE_FILE_SYSTEM = (
    "You are an expert software engineer resolving git merge conflicts. "
    "You will be given a file with conflict markers (<<<<<<< HEAD, =======, >>>>>>>). "
    "Your task is to resolve all conflicts and output ONLY the complete resolved file content. "
    "Do NOT include any explanation, commentary, or markdown formatting. "
    "Output ONLY the resolved file content."
)

WHOLE_FILE_USER_TEMPLATE = """\
Resolve all merge conflicts in the following file.
Output ONLY the resolved file content with all conflicts resolved. No explanation.

File: {file_path}
```{language}
{file_content_with_markers}
```

Resolved file:"""


def build_whole_file_prompt(
    file_path: str,
    file_content_with_markers: str,
) -> list[dict]:
    """Build messages for whole-file conflict resolution."""
    language = _detect_language(file_path)
    return [
        {"role": "system", "content": WHOLE_FILE_SYSTEM},
        {
            "role": "user",
            "content": WHOLE_FILE_USER_TEMPLATE.format(
                file_path=file_path,
                language=language,
                file_content_with_markers=file_content_with_markers,
            ),
        },
    ]


# ---------- Format B: conflict_block ----------

BLOCK_SYSTEM = (
    "You are an expert software engineer resolving a git merge conflict block. "
    "You will be given the conflict block with surrounding context. "
    "Your task is to resolve the conflict and output ONLY the resolved content "
    "that should replace the entire conflict block (from <<<<<<< to >>>>>>>). "
    "Do NOT include conflict markers in your output. "
    "Do NOT include any explanation or markdown formatting. "
    "Output ONLY the resolved code."
)

BLOCK_USER_TEMPLATE = """\
Resolve this git merge conflict block. Output ONLY the resolved content that replaces the conflict block.

File: {file_path}

Context before the conflict:
```{language}
{context_before}
```

Conflict:
```
<<<<<<< HEAD (ours)
{ours_content}=======
{theirs_content}>>>>>>> branch (theirs)
```

Context after the conflict:
```{language}
{context_after}
```

Resolved content:"""


def build_conflict_block_prompt(
    file_path: str,
    block: ConflictBlock,
) -> list[dict]:
    """Build messages for single conflict block resolution."""
    language = _detect_language(file_path)
    return [
        {"role": "system", "content": BLOCK_SYSTEM},
        {
            "role": "user",
            "content": BLOCK_USER_TEMPLATE.format(
                file_path=file_path,
                language=language,
                context_before=block.context_before,
                ours_content=block.ours,
                theirs_content=block.theirs,
                context_after=block.context_after,
            ),
        },
    ]


# ---------- Format C: cherry_pick_whole_file ----------

CHERRY_PICK_SYSTEM = (
    "You are an expert software engineer resolving a git cherry-pick conflict. "
    "A commit is being cherry-picked onto a different branch, causing conflicts. "
    "You will be given the file with conflict markers (<<<<<<< HEAD, =======, >>>>>>>). "
    "HEAD represents the current branch state. The other side represents the "
    "change being cherry-picked. Your task is to resolve all conflicts by "
    "integrating the cherry-picked change into the current branch. "
    "Output ONLY the complete resolved file content. No explanation."
)

CHERRY_PICK_USER_TEMPLATE = """\
Resolve all cherry-pick conflicts in the following file.
A commit is being cherry-picked: "{commit_message}"
Output ONLY the resolved file content. No explanation.

File: {file_path}
```{language}
{file_content_with_markers}
```

Resolved file:"""


def build_cherry_pick_whole_file_prompt(
    file_path: str,
    file_content_with_markers: str,
    commit_message: str = "",
) -> list[dict]:
    """Build messages for cherry-pick whole-file conflict resolution."""
    language = _detect_language(file_path)
    return [
        {"role": "system", "content": CHERRY_PICK_SYSTEM},
        {
            "role": "user",
            "content": CHERRY_PICK_USER_TEMPLATE.format(
                file_path=file_path,
                language=language,
                file_content_with_markers=file_content_with_markers,
                commit_message=commit_message or "N/A",
            ),
        },
    ]


# ---------- Format D: cherry_pick_block ----------

CHERRY_PICK_BLOCK_SYSTEM = (
    "You are an expert software engineer resolving a git cherry-pick conflict block. "
    "A commit is being cherry-picked onto a different branch. "
    "HEAD represents the current branch. The other side is the cherry-picked change. "
    "Resolve the conflict by integrating the cherry-picked change. "
    "Output ONLY the resolved content. No explanation or markers."
)

CHERRY_PICK_BLOCK_USER_TEMPLATE = """\
Resolve this cherry-pick conflict block.
Cherry-picked commit: "{commit_message}"
Output ONLY the resolved content.

File: {file_path}

Context before:
```{language}
{context_before}
```

Conflict:
```
<<<<<<< HEAD (current branch)
{ours_content}=======
{theirs_content}>>>>>>> cherry-picked commit
```

Context after:
```{language}
{context_after}
```

Resolved content:"""


def build_cherry_pick_block_prompt(
    file_path: str,
    block: ConflictBlock,
    commit_message: str = "",
) -> list[dict]:
    """Build messages for cherry-pick single block resolution."""
    language = _detect_language(file_path)
    return [
        {"role": "system", "content": CHERRY_PICK_BLOCK_SYSTEM},
        {
            "role": "user",
            "content": CHERRY_PICK_BLOCK_USER_TEMPLATE.format(
                file_path=file_path,
                language=language,
                context_before=block.context_before,
                ours_content=block.ours,
                theirs_content=block.theirs,
                context_after=block.context_after,
                commit_message=commit_message or "N/A",
            ),
        },
    ]
