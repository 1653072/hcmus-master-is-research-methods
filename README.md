# HCMUS Master IS - Research Methods Course
**Description**: Repository for Research Methods course, used to store the codebase of AI models


## Quick Notes

1. There are 3 large files exceeding **100MB**, so we need to push them to Github LFS (i.e., Large File Storage):

    - hfgat_rewrite_validate/hfgat_rewrite_validate/output_hfgat_notebook/cache/item_features.pt
    - hfgat_rewrite_validate/hfgat_rewrite_validate/output_hfgat_notebook/cache_old/item_features.pt
    - hfgat_rewrite_validate/hfgat_rewrite_validate/output_hfgat_notebook/cache_old1/item_features.pt

2. Install Git LFS:

    - MacOS: `brew install git-lfs` or `sudo apt install git-lfs`
    - Windows: `choco install git-lfs`
    - Finally, install LFS into the current repository: `git lfs install`

3. Track/Untrack large files

    - Track: `git lfs track "*.pt"`
    - Untrack: `git lfs untrack "*.pt"`

4. Pull latest LFS changes:

    - Command: `git lfs fetch --all && git lfs pull`

5. De-duplicate LFS files to save disk:

    - Command: `git lfs dedup`

6. Check reference links and environment of LFS files:

    - Comman: `git lfs env`

