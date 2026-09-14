<#
.SYNOPSIS
    Links scientific research agent skills and project skills for Google Antigravity.

.DESCRIPTION
    Recreates the directory junctions in .agents/skills and .agents/plugins/scientific-research-agent/skills
    matching the skills defined in opencode.json.
#>

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$externalSkillsDir = Join-Path $HOME ".config\opencode\external\scientific-agent-skills\skills"
$projectSkillsDir = Join-Path $repoRoot "tools\opencode\skills"

$curatedSkills = @(
    "pytorch-lightning", "scikit-learn", "statistical-analysis", "matplotlib", "seaborn",
    "scientific-visualization", "optimize-for-gpu", "get-available-resources", "pathml",
    "histolab", "pydicom", "scientific-writing", "literature-review", "paper-lookup",
    "research-lookup", "bgpt-paper-search", "citation-management", "peer-review",
    "markdown-mermaid-writing", "docx", "pptx", "scientific-slides", "venue-templates",
    "latex-posters", "exploratory-data-analysis", "hypothesis-generation", "experimental-design",
    "exa-search", "parallel-web", "hugging-science", "markitdown"
)

$targetDirs = @(
    (Join-Path $repoRoot ".agents\skills"),
    (Join-Path $repoRoot ".agents\plugins\scientific-research-agent\skills")
)

foreach ($target in $targetDirs) {
    if (-not (Test-Path $target)) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
    }

    # 1. Link project local skill (pdf-translate)
    $localSkillSrc = Join-Path $projectSkillsDir "pdf-translate"
    $localSkillDst = Join-Path $target "pdf-translate"
    if (Test-Path $localSkillSrc) {
        if (-not (Test-Path $localSkillDst)) {
            New-Item -ItemType Junction -Path $localSkillDst -Target $localSkillSrc | Out-Null
            Write-Host "Linked local skill: pdf-translate -> $target"
        }
    }

    # 2. Link external scientific skills
    foreach ($skill in $curatedSkills) {
        $srcPath = Join-Path $externalSkillsDir $skill
        $dstPath = Join-Path $target $skill
        if (Test-Path $srcPath) {
            if (-not (Test-Path $dstPath)) {
                New-Item -ItemType Junction -Path $dstPath -Target $srcPath | Out-Null
                Write-Host "Linked external skill: $skill -> $target"
            }
        } else {
            Write-Warning "External skill not found: $srcPath"
        }
    }
}

# 3. Validate via agy CLI if available
if (Get-Command agy.exe -ErrorAction SilentlyContinue) {
    Write-Host "`nValidating Antigravity plugin..."
    $pluginPath = Join-Path $repoRoot ".agents\plugins\scientific-research-agent"
    & agy.exe plugin validate $pluginPath
}

Write-Host "`nAntigravity Scientific Research Skills setup complete!"
