#!/usr/bin/env python3
"""
BlackRoad Image Optimizer - Image optimization, resizing, and CDN pipeline
"""

import sqlite3
import uuid
import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    WEBP = "webp"
    AVIF = "avif"
    PNG = "png"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Format compression ratios relative to JPEG
FORMAT_COMPRESSION: Dict[str, float] = {
    "jpeg": 1.0,
    "webp": 0.65,
    "avif": 0.50,
    "png": 1.8,
}

# Quality multipliers
QUALITY_FACTOR: Dict[int, float] = {
    60: 0.4, 70: 0.55, 75: 0.65, 80: 0.75, 85: 0.85, 90: 1.0, 95: 1.3
}


@dataclass
class ImageVariant:
    id: str
    job_id: str
    name: str
    width: int
    height: int
    format: str
    quality: int
    size_bytes: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


@dataclass
class CDNEntry:
    id: str
    variant_id: str
    cdn_url: str
    cache_ttl: int  # seconds
    hits: int = 0
    purged: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_hit: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizeJob:
    id: str
    source_url: str
    status: str = JobStatus.PENDING.value
    variant_specs: str = "[]"  # JSON
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    total_input_bytes: int = 0
    total_output_bytes: int = 0

    @property
    def variants_list(self) -> list:
        return json.loads(self.variant_specs or "[]")

    @property
    def savings_percent(self) -> float:
        if self.total_input_bytes == 0:
            return 0.0
        saved = self.total_input_bytes - self.total_output_bytes
        return round((saved / self.total_input_bytes) * 100, 2)

    def to_dict(self) -> dict:
        return asdict(self)


class ImageOptimizer:
    """Image optimization, resizing, and CDN pipeline manager."""

    CDN_BASE = "https://cdn.blackroad.io/images"
    DEFAULT_CACHE_TTL = 86400 * 30  # 30 days

    RESPONSIVE_VARIANTS = [
        {"name": "mobile", "width": 480, "height": 320, "format": "webp", "quality": 80},
        {"name": "tablet", "width": 768, "height": 512, "format": "webp", "quality": 82},
        {"name": "desktop", "width": 1200, "height": 800, "format": "webp", "quality": 85},
        {"name": "retina", "width": 2400, "height": 1600, "format": "webp", "quality": 85},
        {"name": "mobile-jpeg", "width": 480, "height": 320, "format": "jpeg", "quality": 75},
        {"name": "desktop-avif", "width": 1200, "height": 800, "format": "avif", "quality": 80},
    ]

    def __init__(self, db_path: str = "image_optimizer.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS optimize_jobs (
                    id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    variant_specs TEXT DEFAULT '[]',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_input_bytes INTEGER DEFAULT 0,
                    total_output_bytes INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS image_variants (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    quality INTEGER NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES optimize_jobs(id)
                );

                CREATE TABLE IF NOT EXISTS cdn_entries (
                    id TEXT PRIMARY KEY,
                    variant_id TEXT NOT NULL,
                    cdn_url TEXT NOT NULL,
                    cache_ttl INTEGER NOT NULL,
                    hits INTEGER DEFAULT 0,
                    purged INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_hit TEXT,
                    FOREIGN KEY (variant_id) REFERENCES image_variants(id)
                );
            """)

    def _simulate_source_size(self, url: str) -> int:
        """Simulate source image size based on URL hash."""
        h = int(hashlib.md5(url.encode()).hexdigest(), 16)
        return (h % 4_000_000) + 500_000  # 500KB - 4.5MB

    def _simulate_source_dimensions(self, url: str) -> Tuple[int, int]:
        """Simulate source image dimensions."""
        h = int(hashlib.sha256(url.encode()).hexdigest(), 16)
        widths = [800, 1200, 1600, 1920, 2400, 3000, 4000]
        w = widths[h % len(widths)]
        aspect = 16 / 9 if (h >> 8) % 2 == 0 else 4 / 3
        return w, int(w / aspect)

    def _calculate_output_size(self, source_bytes: int, src_w: int, src_h: int,
                                target_w: int, target_h: int, fmt: str, quality: int) -> int:
        """Calculate output file size using area ratio and format compression."""
        area_ratio = (target_w * target_h) / max(src_w * src_h, 1)
        fmt_ratio = FORMAT_COMPRESSION.get(fmt, 1.0)
        # Find nearest quality factor
        nearest_q = min(QUALITY_FACTOR.keys(), key=lambda k: abs(k - quality))
        q_factor = QUALITY_FACTOR[nearest_q]
        return max(1024, int(source_bytes * area_ratio * fmt_ratio * q_factor))

    def _build_cdn_url(self, job_id: str, variant_name: str, fmt: str) -> str:
        return f"{self.CDN_BASE}/{job_id}/{variant_name}.{fmt}"

    def submit(self, source_url: str, variants: List[dict]) -> OptimizeJob:
        """Submit an image optimization job."""
        job_id = str(uuid.uuid4())
        job = OptimizeJob(
            id=job_id,
            source_url=source_url,
            variant_specs=json.dumps(variants),
        )
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO optimize_jobs
                (id, source_url, status, variant_specs, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (job.id, job.source_url, job.status,
                  job.variant_specs, job.created_at))
        return job

    def process(self, job_id: str) -> OptimizeJob:
        """Process all variants for an optimization job (resize via math, no PIL)."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status == JobStatus.COMPLETED.value:
            return job

        with self._connect() as conn:
            conn.execute("UPDATE optimize_jobs SET status='processing' WHERE id=?", (job_id,))

        try:
            source_bytes = self._simulate_source_size(job.source_url)
            src_w, src_h = self._simulate_source_dimensions(job.source_url)
            variants = job.variants_list
            total_output = 0

            for spec in variants:
                variant_id = str(uuid.uuid4())
                target_w = spec.get("width", 800)
                target_h = spec.get("height", int(target_w * src_h / max(src_w, 1)))
                fmt = spec.get("format", "webp")
                quality = spec.get("quality", 85)
                name = spec.get("name", f"{target_w}x{target_h}")

                out_size = self._calculate_output_size(
                    source_bytes, src_w, src_h, target_w, target_h, fmt, quality
                )
                total_output += out_size

                with self._connect() as conn:
                    conn.execute("""
                        INSERT INTO image_variants
                        (id, job_id, name, width, height, format, quality, size_bytes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (variant_id, job_id, name, target_w, target_h,
                          fmt, quality, out_size, datetime.now(timezone.utc).isoformat()))

                    cdn_id = str(uuid.uuid4())
                    cdn_url = self._build_cdn_url(job_id, name, fmt)
                    conn.execute("""
                        INSERT INTO cdn_entries
                        (id, variant_id, cdn_url, cache_ttl, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (cdn_id, variant_id, cdn_url, self.DEFAULT_CACHE_TTL,
                          datetime.now(timezone.utc).isoformat()))

            completed_at = datetime.now(timezone.utc).isoformat()
            with self._connect() as conn:
                conn.execute("""
                    UPDATE optimize_jobs
                    SET status='completed', completed_at=?,
                        total_input_bytes=?, total_output_bytes=?
                    WHERE id=?
                """, (completed_at, source_bytes, total_output, job_id))

        except Exception as e:
            with self._connect() as conn:
                conn.execute("""
                    UPDATE optimize_jobs SET status='failed', error_message=? WHERE id=?
                """, (str(e), job_id))

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[OptimizeJob]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM optimize_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        return OptimizeJob(**dict(row))

    def get_variants(self, job_id: str) -> List[ImageVariant]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_variants WHERE job_id=? ORDER BY width", (job_id,)
            ).fetchall()
        return [ImageVariant(**dict(r)) for r in rows]

    def generate_srcset(self, job_id: str) -> str:
        """Generate HTML srcset attribute string for a job's variants."""
        variants = self.get_variants(job_id)
        webp_variants = [v for v in variants if v.format == "webp"]
        if not webp_variants:
            webp_variants = variants  # fallback to all
        entries = []
        for v in sorted(webp_variants, key=lambda x: x.width):
            cdn = self._get_cdn_for_variant(v.id)
            if cdn:
                entries.append(f"{cdn.cdn_url} {v.width}w")
        return ", ".join(entries)

    def _get_cdn_for_variant(self, variant_id: str) -> Optional[CDNEntry]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cdn_entries WHERE variant_id=? AND purged=0", (variant_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["purged"] = bool(d["purged"])
        return CDNEntry(**d)

    def purge_cdn(self, variant_id: str) -> bool:
        """Purge a CDN entry (mark as purged)."""
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE cdn_entries SET purged=1 WHERE variant_id=?", (variant_id,)
            )
        return result.rowcount > 0

    def record_cdn_hit(self, variant_id: str) -> bool:
        """Record a CDN cache hit."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            result = conn.execute("""
                UPDATE cdn_entries SET hits=hits+1, last_hit=?
                WHERE variant_id=? AND purged=0
            """, (now, variant_id))
        return result.rowcount > 0

    def get_metrics(self, job_id: str) -> dict:
        """Get optimization metrics for a job."""
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        variants = self.get_variants(job_id)
        cdn_hits = 0
        with self._connect() as conn:
            for v in variants:
                row = conn.execute(
                    "SELECT SUM(hits) FROM cdn_entries WHERE variant_id=?", (v.id,)
                ).fetchone()
                cdn_hits += row[0] or 0

        return {
            "job_id": job_id,
            "status": job.status,
            "source_url": job.source_url,
            "total_input_bytes": job.total_input_bytes,
            "total_output_bytes": job.total_output_bytes,
            "savings_percent": job.savings_percent,
            "variant_count": len(variants),
            "cdn_hits": cdn_hits,
            "formats": list({v.format for v in variants}),
        }

    def create_responsive_set(self, url: str) -> OptimizeJob:
        """Create mobile/tablet/desktop/retina variant set for a URL."""
        return self.submit(url, self.RESPONSIVE_VARIANTS)

    def optimize_batch(self, urls: List[str],
                       variant_specs: Optional[List[dict]] = None) -> List[dict]:
        """Submit and process multiple images."""
        specs = variant_specs or self.RESPONSIVE_VARIANTS
        results = []
        for url in urls:
            job = self.submit(url, specs)
            completed = self.process(job.id)
            results.append({
                "url": url,
                "job_id": completed.id,
                "status": completed.status,
                "savings_percent": completed.savings_percent,
            })
        return results

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[OptimizeJob]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM optimize_jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM optimize_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [OptimizeJob(**dict(r)) for r in rows]

    def get_cdn_stats(self) -> dict:
        """Get CDN statistics across all variants."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cdn_entries").fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hits) FROM cdn_entries").fetchone()[0] or 0
            purged = conn.execute("SELECT COUNT(*) FROM cdn_entries WHERE purged=1").fetchone()[0]
        return {
            "total_cdn_entries": total,
            "total_hits": total_hits,
            "purged_entries": purged,
            "active_entries": total - purged,
        }

    def get_format_breakdown(self, job_id: str) -> dict:
        """Get size breakdown by format for a job."""
        variants = self.get_variants(job_id)
        breakdown: Dict[str, dict] = {}
        for v in variants:
            if v.format not in breakdown:
                breakdown[v.format] = {"count": 0, "total_bytes": 0}
            breakdown[v.format]["count"] += 1
            breakdown[v.format]["total_bytes"] += v.size_bytes
        return breakdown

    def estimate_savings(self, source_url: str, target_format: str = "webp",
                         target_quality: int = 85) -> dict:
        """Estimate savings for a URL without processing."""
        source_bytes = self._simulate_source_size(source_url)
        src_w, src_h = self._simulate_source_dimensions(source_url)
        out_bytes = self._calculate_output_size(
            source_bytes, src_w, src_h, src_w, src_h, target_format, target_quality
        )
        return {
            "source_url": source_url,
            "estimated_source_bytes": source_bytes,
            "estimated_output_bytes": out_bytes,
            "estimated_savings_percent": round((1 - out_bytes / source_bytes) * 100, 2),
            "target_format": target_format,
            "source_dimensions": f"{src_w}x{src_h}",
        }


def create_optimizer(db_path: str = "image_optimizer.db") -> ImageOptimizer:
    return ImageOptimizer(db_path=db_path)


if __name__ == "__main__":
    optimizer = create_optimizer()

    print("BlackRoad Image Optimizer")
    print("=" * 40)

    # Responsive set
    job = optimizer.create_responsive_set("https://example.com/hero.jpg")
    processed = optimizer.process(job.id)
    print(f"Processed job: {processed.id}")
    print(f"Status: {processed.status}")
    print(f"Savings: {processed.savings_percent}%")

    variants = optimizer.get_variants(job.id)
    print(f"Variants created: {len(variants)}")
    for v in variants:
        print(f"  {v.name}: {v.width}x{v.height} {v.format} q{v.quality} = {v.size_bytes//1024}KB")

    srcset = optimizer.generate_srcset(job.id)
    print(f"\nsrcset (first 100 chars): {srcset[:100]}...")

    metrics = optimizer.get_metrics(job.id)
    print(f"\nMetrics: {metrics}")

    batch = optimizer.optimize_batch([
        "https://example.com/image1.jpg",
        "https://example.com/image2.png",
    ])
    print(f"\nBatch results: {batch}")

    estimate = optimizer.estimate_savings("https://example.com/large.jpg", "avif", 80)
    print(f"\nEstimated savings to AVIF: {estimate['estimated_savings_percent']}%")
