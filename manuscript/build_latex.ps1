[CmdletBinding()]
param(
    [string]$MainFile = "main.tex",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Resolve-Executable([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Required executable '$Name' was not found on PATH. Install TeX Live/MiKTeX or add its bin directory to PATH."
    }
    return $command.Source
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    Write-Host (">> " + $Executable + " " + ($Arguments -join " ")) -ForegroundColor DarkGray
    # Keep intermediate LaTeX diagnostics out of the terminal.  The first
    # pdflatex pass can legitimately report undefined citations before BibTeX
    # has populated the bibliography.  Print the captured output only when a
    # command actually fails.
    $output = @(& $Executable @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $output | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        throw "Command failed with exit code ${exitCode}: $Executable"
    }
}

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$mainPath = Join-Path $scriptDirectory $MainFile
if (-not (Test-Path -LiteralPath $mainPath -PathType Leaf)) {
    throw "Main TeX file not found: $mainPath"
}

$pdflatex = Resolve-Executable "pdflatex"
$bibtex = Resolve-Executable "bibtex"
$mainStem = [System.IO.Path]::GetFileNameWithoutExtension($MainFile)

if ($Clean) {
    $generatedExtensions = @(
        ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log",
        ".lof", ".lot", ".out", ".synctex.gz", ".toc"
    )
    foreach ($extension in $generatedExtensions) {
        $generatedPath = Join-Path $scriptDirectory ($mainStem + $extension)
        Remove-Item -LiteralPath $generatedPath -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Removed generated LaTeX auxiliary files for a clean build." -ForegroundColor DarkYellow
}

# A stale .bbl can contain bytes produced by an earlier, differently encoded
# .bib file.  In that case pdfTeX fails before BibTeX gets a chance to rebuild
# the bibliography.  Invalidate only the generated bibliography when the
# source bibliography is newer; manuscript source and figures are untouched.
$bibSource = Join-Path $scriptDirectory "references.bib"
$bblPath = Join-Path $scriptDirectory ($mainStem + ".bbl")
$blgPath = Join-Path $scriptDirectory ($mainStem + ".blg")
if (Test-Path -LiteralPath $bibSource -PathType Leaf) {
    $bibLines = @(Get-Content -LiteralPath $bibSource -Encoding UTF8)
    $bibKeys = @(
        $bibLines |
            Select-String -Pattern '^\s*@[A-Za-z]+\s*\{\s*([^,]+),' |
            ForEach-Object { $_.Matches[0].Groups[1].Value.Trim() }
    )
    $duplicateBibKeys = @(
        $bibKeys |
            Group-Object |
            Where-Object { $_.Count -gt 1 } |
            ForEach-Object { $_.Name }
    )
    if ($duplicateBibKeys.Count -gt 0) {
        throw ("Duplicate BibTeX keys in {0}: {1}" -f $bibSource, ($duplicateBibKeys -join ', '))
    }

    # This pdfLaTeX/BibTeX workflow is deliberately ASCII-safe. Accented
    # bibliography text should use TeX escapes, for example Ca{\~n}adas.
    $nonAsciiBibLines = @(
        Select-String -LiteralPath $bibSource -Pattern '[^\x00-\x7F]' -Encoding UTF8
    )
    if ($nonAsciiBibLines.Count -gt 0) {
        $lineNumbers = $nonAsciiBibLines.LineNumber -join ', '
        throw ("Non-ASCII bibliography text in {0} at line(s) {1}. Use TeX accent escapes." -f $bibSource, $lineNumbers)
    }
}

if ((Test-Path -LiteralPath $bibSource -PathType Leaf) -and
    (Test-Path -LiteralPath $bblPath -PathType Leaf)) {
    $bibTime = (Get-Item -LiteralPath $bibSource).LastWriteTimeUtc
    $bblTime = (Get-Item -LiteralPath $bblPath).LastWriteTimeUtc
    if ($bibTime -gt $bblTime) {
        Remove-Item -LiteralPath @($bblPath, $blgPath) -Force -ErrorAction SilentlyContinue
        Write-Host "Bibliography source is newer; stale .bbl/.blg removed." -ForegroundColor DarkYellow
    }
}

$texArguments = @(
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-file-line-error",
    $MainFile
)

Push-Location $scriptDirectory
try {
    # Run from the manuscript directory so relative \input, figures, .bst,
    # and .bib paths resolve identically on every machine.
    Invoke-Checked $pdflatex $texArguments
    Invoke-Checked $bibtex @($mainStem)
    Invoke-Checked $pdflatex $texArguments
    Invoke-Checked $pdflatex $texArguments

    $pdfPath = Join-Path $scriptDirectory ($mainStem + ".pdf")
    $logPath = Join-Path $scriptDirectory ($mainStem + ".log")
    if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
        throw "Compilation returned successfully but the PDF was not produced: $pdfPath"
    }

    # A zero exit code can still hide unresolved cross-references. Treat those
    # as failures for a reproducible manuscript build.
    $fatalPatterns = @(
        '(?m)^!',
        'Emergency stop',
        'Fatal error',
        'Undefined control sequence'
    )
    $referencePatterns = @(
        'Citation .* undefined',
        'Reference .* undefined',
        'There were undefined citations',
        'There were undefined references'
    )
    $fatalMatches = @()
    $referenceMatches = @()
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $fatalMatches = @(Select-String -LiteralPath $logPath -Pattern $fatalPatterns)
        $referenceMatches = @(Select-String -LiteralPath $logPath -Pattern $referencePatterns)
    }
    if ($fatalMatches.Count -gt 0) {
        throw "Fatal LaTeX diagnostics remain in $logPath."
    }
    if ($referenceMatches.Count -gt 0) {
        $referenceMatches | ForEach-Object { Write-Host $_.Line -ForegroundColor Red }
        throw "Undefined citations or references remain in $logPath."
    }

    $overfullCount = 0
    $duplicateDestinationCount = 0
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $overfullCount = @(Select-String -LiteralPath $logPath -Pattern 'Overfull \\hbox').Count
        $duplicateDestinationCount = @(Select-String -LiteralPath $logPath -Pattern 'destination with the same identifier').Count
    }

    $pdf = Get-Item -LiteralPath $pdfPath
    Write-Host "SUCCESS: $($pdf.FullName)" -ForegroundColor Green
    Write-Host ("PDF size: {0:N0} bytes" -f $pdf.Length)
    if ($overfullCount -gt 0 -or $duplicateDestinationCount -gt 0) {
        Write-Host "Non-fatal layout warnings: Overfull hbox=$overfullCount; duplicate PDF destinations=$duplicateDestinationCount" -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}
