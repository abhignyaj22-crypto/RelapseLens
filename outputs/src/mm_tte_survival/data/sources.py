"""Public-only, verifiable data acquisition sources (Stage B).

Controlled-access sources (dbGaP raw, EGA, BeatAML raw, CoMMpass PFS variants)
raise at construction — never referenced. Public sources download their artifact
idempotently and expose a checksum/rowcount/schema audit via ``verify()``.

Design note: the controlled-access guard is enforced two ways —
  * ``__init_subclass__`` refuses to even *define* a source whose ``source_id``
    contains a controlled marker, and
  * :func:`assert_public` refuses a controlled identifier at run time (e.g. when a
    config references one).

URL provenance (resolved + reachability-checked in Stage B):
  * STRING v12 ....... stringdb-downloads.org (CC BY 4.0)
  * CCLE proteomics .. gygi.hms.harvard.edu mirror (Nusinow 2020) — DepMap portal is
    now behind a Cloudflare challenge, so the Gygi lab mirror is used.
  * CCLE epigenomics . data.broadinstitute.org/ccle Global Chromatin Profiling (histone
    PTM). RRBS methylation mirror is unresolved on the Broad bucket → documented GAP.
  * DepMap CRISPR .... ndownloader.figshare.com (DepMap 23Q2 Public, bypasses Cloudflare)
  * GDSC2 ............ cog.sanger.ac.uk (cancerrxgene release 8.5)
  * Reactome ......... reactome.org/download/current (CC0)
  * GEO .............. ftp.ncbi.nlm.nih.gov (series matrix for microarray; RAW/suppl for
    RNA-Seq + scRNA, whose counts are not in the series matrix)
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

# Controlled identifiers rejected at construction / resolution
_CONTROLLED_MARKERS = (
    "dbgap", "phs000748", "controlled", "ega", "beataml_raw",
    "pfs_controlled", "researcher gateway raw",
)


def assert_public(source_id: str) -> None:
    """Raise if ``source_id`` names a controlled-access tier. Public-data-only guard."""
    low = (source_id or "").lower()
    if any(m in low for m in _CONTROLLED_MARKERS):
        raise ValueError(f"Controlled source rejected: {source_id}")


@dataclass
class DataAudit:
    source_id: str
    path: Path
    rows: int | None = None
    cols: int | None = None
    checksum: str | None = None
    schema_ok: bool = True
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.schema_ok and self.path.exists()


class DataSource(ABC):
    """Verifiable public data source."""

    source_id: ClassVar[str]
    is_public: ClassVar[bool] = True
    modality: ClassVar[str]
    domain: ClassVar[str]  # patient | cell_line | network

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sid = getattr(cls, "source_id", "") or ""
        low = sid.lower()
        if any(m in low for m in _CONTROLLED_MARKERS):
            raise ValueError(f"Controlled source rejected: {sid}")

    @abstractmethod
    def fetch(self, dest: Path) -> Path:
        """Download or materialize data at dest. Idempotent."""

    @abstractmethod
    def verify(self) -> DataAudit:
        """Checksum/rowcount/schema verification."""

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _md5(path: Path, n: int = 65536) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(n):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _download(url: str, out: Path, *, min_bytes: int = 1, timeout: int = 3600) -> Path:
        """curl download (follow redirects, resume, fail loud). Idempotent: skips when
        the target already exists and meets ``min_bytes``."""
        if out.exists() and out.stat().st_size >= min_bytes:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        subprocess.run(
            ["curl", "-fSL", "--retry", "3", "--retry-delay", "5", "-C", "-",
             "--max-time", str(timeout), "-o", str(tmp), url],
            check=True,
        )
        if min_bytes > 1 and tmp.stat().st_size < min_bytes:
            raise RuntimeError(f"{url} -> {tmp.stat().st_size} bytes < expected {min_bytes} "
                               f"(likely a redirect/challenge page, not the data)")
        tmp.rename(out)
        return out

    @staticmethod
    def _list_dir(url: str) -> list[str]:
        """List hrefs under an HTTP(S) directory (GEO FTP-over-HTTPS)."""
        html = subprocess.run(["curl", "-fsSL", "--max-time", "120", url],
                              capture_output=True, text=True, check=True).stdout
        return sorted(set(re.findall(r'href="([^"?]+)"', html)))

    def _audit_file(self, path: Path, *, notes: str = "") -> DataAudit:
        if not path.exists():
            return DataAudit(self.source_id, path, schema_ok=False, notes=notes or "not fetched")
        return DataAudit(self.source_id, path, checksum=self._md5(path),
                         schema_ok=path.stat().st_size > 0, notes=notes)


class GDCCoMMpassOpen(DataSource):
    """Patient transcriptomics/clinical are already on disk (data/real, MMRF-CoMMpass
    OPEN). This source verifies presence rather than downloading — the open GDC pull is
    handled by scripts/realdata/fetch_{clinical,rna,cnv}.py."""

    source_id = "gdc_commpass_open"
    modality = "transcriptomics"
    domain = "patient"

    def __init__(self, data_root: Path | None = None):
        self.data_root = Path(data_root or "data/real")

    def fetch(self, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def verify(self) -> DataAudit:
        clin = self.data_root / "clinical_survival.csv"
        if not clin.exists():
            return DataAudit(self.source_id, clin, schema_ok=False, notes="clinical_survival.csv missing")
        import pandas as pd
        df = pd.read_csv(clin, nrows=0)
        return DataAudit(self.source_id, clin, cols=len(df.columns), checksum=self._md5(clin))


class CCLEProteomics(DataSource):
    source_id = "ccle_proteomics_gygi"
    modality = "proteomics"
    domain = "cell_line"
    URL = "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz"
    FILENAME = "protein_quant_current_normalized.csv.gz"

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/proteomics_cellline")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        return self._download(self.URL, d / self.FILENAME, min_bytes=10_000_000)

    def verify(self) -> DataAudit:
        return self._audit_file(self.dest / self.FILENAME,
                                notes="SURROGATE(cell_line) — not patient proteomics")


class CCLEEpigenomics(DataSource):
    source_id = "ccle_epigenomics_depmap"
    modality = "epigenomics"
    domain = "cell_line"
    # Histone-PTM global chromatin profiling (Broad CCLE bucket). RRBS methylation
    # mirror is currently unresolved on the Broad bucket -> documented GAP, not faked.
    HISTONE_URL = "https://data.broadinstitute.org/ccle/CCLE_GlobalChromatinProfiling_20181130.csv"
    HISTONE_FILE = "CCLE_GlobalChromatinProfiling_20181130.csv"

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/epigenomics_cellline")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        out = self._download(self.HISTONE_URL, d / self.HISTONE_FILE, min_bytes=100_000)
        (d / "GAP_RRBS_and_ATAC.md").write_text(
            "# Epigenomics gaps (public-data-only)\n\n"
            "* Histone-PTM (global chromatin profiling): PRESENT (this folder).\n"
            "* CCLE RRBS methylation: Broad bucket mirror unresolved -> GAP (revisit).\n"
            "* Patient ATAC-seq: does NOT exist publicly for MM -> GAP by construction.\n"
        )
        return out

    def verify(self) -> DataAudit:
        return self._audit_file(self.dest / self.HISTONE_FILE,
                                notes="SURROGATE(cell_line); histone-PTM present; RRBS+ATAC gaps documented")


class StringV12(DataSource):
    source_id = "string_v12_9606"
    modality = "ppi"
    domain = "network"
    BASE = "https://stringdb-downloads.org/download"
    FILES = {
        "9606.protein.links.full.v12.0.txt.gz": (f"{BASE}/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz", 10_000_000),
        "9606.protein.info.v12.0.txt.gz": (f"{BASE}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz", 100_000),
        "9606.protein.aliases.v12.0.txt.gz": (f"{BASE}/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz", 1_000_000),
    }

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/ppi")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        for fn, (url, mb) in self.FILES.items():
            self._download(url, d / fn, min_bytes=mb)
        return d / "9606.protein.links.full.v12.0.txt.gz"

    def verify(self) -> DataAudit:
        links = self.dest / "9606.protein.links.full.v12.0.txt.gz"
        missing = [fn for fn in self.FILES if not (self.dest / fn).exists()]
        return DataAudit(self.source_id, links, schema_ok=not missing,
                         checksum=self._md5(links) if links.exists() else None,
                         notes="CC BY 4.0" + (f"; MISSING {missing}" if missing else ""))


class GDSC2DoseResponse(DataSource):
    source_id = "gdsc2_dose_response"
    modality = "drug_response"
    domain = "cell_line"
    URL = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
    FILENAME = "GDSC2_fitted_dose_response_27Oct23.xlsx"

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/drug_response")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        return self._download(self.URL, d / self.FILENAME, min_bytes=1_000_000)

    def verify(self) -> DataAudit:
        return self._audit_file(self.dest / self.FILENAME, notes="GDSC2 release 8.5")


class DepMapCRISPR(DataSource):
    source_id = "depmap_crispr_23q2"
    modality = "drug_response"  # CRISPR gene-effect (resistance-gene prior)
    domain = "cell_line"
    # DepMap 23Q2 Public via figshare (bypasses depmap.org Cloudflare challenge)
    FILES = {
        "CRISPRGeneEffect.csv": ("https://ndownloader.figshare.com/files/40448555", 10_000_000),
        "Model.csv": ("https://ndownloader.figshare.com/files/40448834", 100_000),
    }

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/drug_response")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        for fn, (url, mb) in self.FILES.items():
            self._download(url, d / fn, min_bytes=mb)
        return d / "CRISPRGeneEffect.csv"

    def verify(self) -> DataAudit:
        return self._audit_file(self.dest / "CRISPRGeneEffect.csv", notes="DepMap 23Q2 Public (figshare)")


class ReactomePPI(DataSource):
    source_id = "reactome_interactions"
    modality = "ppi"
    domain = "network"
    URL = "https://reactome.org/download/current/interactors/reactome.homo_sapiens.interactions.tab-delimited.txt"
    FILENAME = "reactome.homo_sapiens.interactions.tab-delimited.txt"

    def __init__(self, dest: Path | None = None):
        self.dest = Path(dest or "data/external/ppi")

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        return self._download(self.URL, d / self.FILENAME, min_bytes=100_000)

    def verify(self) -> DataAudit:
        return self._audit_file(self.dest / self.FILENAME, notes="Reactome (CC0)")


class GEOSeries(DataSource):
    """GEO fetcher. Microarray series (GSE9782) carry expression in the series matrix;
    RNA-Seq / scRNA series (GSE116324, GSE271107, GSE124310) carry only metadata in the
    matrix, so their processed counts are pulled from the suppl/ RAW files."""

    source_id: ClassVar[str]
    gse_id: str
    modality: ClassVar[str] = "transcriptomics"
    domain: ClassVar[str] = "patient"

    PROVENANCE: ClassVar[dict[str, str]] = {
        "GSE124310": "Zavidij/Ghobrial, Nat Cancer 2020 (NOT Ledergor)",
        "GSE117156": "Ledergor 2018 (verify before citing)",
        "GSE9782": "Mulligan 2007 bortezomib response (APEX)",
        "GSE116324": "PADIMAC (RNA-Seq; counts in suppl)",
        "GSE271107": "scRNA open (GSE271107_RAW.tar in suppl)",
    }
    # which series also need supplementary RAW files (not just the series matrix)
    NEEDS_SUPPL: ClassVar[set[str]] = {"GSE116324", "GSE271107", "GSE124310"}

    def __init__(self, gse_id: str, dest: Path | None = None):
        assert_public(gse_id)
        self.gse_id = gse_id.upper()
        self.source_id = f"geo_{self.gse_id.lower()}"
        self.dest = Path(dest or f"data/external/drug_labeled/{self.gse_id}")

    def _stub(self) -> str:
        n = self.gse_id[3:]
        return f"GSE{n[:-3]}nnn" if len(n) > 3 else "GSEnnn"

    def _base(self, kind: str) -> str:
        return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{self._stub()}/{self.gse_id}/{kind}/"

    def fetch(self, dest: Path | None = None) -> Path:
        d = Path(dest or self.dest)
        d.mkdir(parents=True, exist_ok=True)
        # 1) series matrix (metadata always; expression for microarray)
        for href in self._list_dir(self._base("matrix")):
            fn = href.split("/")[-1]
            if fn.endswith("_series_matrix.txt.gz"):
                self._download(self._base("matrix") + fn, d / fn, min_bytes=1000)
        # 2) supplementary RAW for RNA-Seq / scRNA
        last = None
        if self.gse_id in self.NEEDS_SUPPL:
            for href in self._list_dir(self._base("suppl")):
                fn = href.split("/")[-1]
                if fn.startswith(self.gse_id) and (fn.endswith(".tar") or fn.endswith(".gz")
                                                   or fn.endswith(".txt") or fn.endswith(".csv")):
                    last = self._download(self._base("suppl") + fn, d / fn, min_bytes=1000, timeout=7200)
        return last or next(iter(d.glob("*_series_matrix.txt.gz")), d)

    def verify(self) -> DataAudit:
        hits = sorted(self.dest.glob("*"))
        files = [p for p in hits if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        return DataAudit(self.source_id, self.dest, rows=len(files),
                         schema_ok=bool(files) and total > 0,
                         notes=f"{self.PROVENANCE.get(self.gse_id, '')}; {len(files)} file(s), {total/1e6:.1f} MB")


# Registry for orchestrator / config resolution
PUBLIC_SOURCES: dict[str, type[DataSource]] = {
    "gdc_commpass_open": GDCCoMMpassOpen,
    "ccle_proteomics_gygi": CCLEProteomics,
    "ccle_epigenomics_depmap": CCLEEpigenomics,
    "string_v12_9606": StringV12,
    "gdsc2_dose_response": GDSC2DoseResponse,
    "depmap_crispr_23q2": DepMapCRISPR,
    "reactome_interactions": ReactomePPI,
}

# GEO series (instantiated with an id, not class-registered)
GEO_SERIES: dict[str, str] = {
    "GSE9782": "data/external/drug_labeled/GSE9782",
    "GSE116324": "data/external/drug_labeled/GSE116324",
    "GSE271107": "data/external/scrna/GSE271107",
    "GSE124310": "data/external/scrna/GSE124310",
}
