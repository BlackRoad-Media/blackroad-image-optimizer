import pytest
from image_optimizer import (
    ImageOptimizer, OptimizeJob, ImageVariant, CDNEntry,
    ImageFormat, JobStatus, FORMAT_COMPRESSION,
    create_optimizer,
)


@pytest.fixture
def optimizer(tmp_path):
    return ImageOptimizer(db_path=str(tmp_path / "test_optimizer.db"))


class TestJobSubmission:
    def test_submit_job(self, optimizer):
        job = optimizer.submit("https://example.com/img.jpg", [
            {"name": "thumb", "width": 200, "height": 200, "format": "webp", "quality": 80}
        ])
        assert job.id is not None
        assert job.status == JobStatus.PENDING.value
        assert job.source_url == "https://example.com/img.jpg"

    def test_submit_stores_variants_spec(self, optimizer):
        specs = [{"name": "sm", "width": 300, "height": 200, "format": "jpeg", "quality": 85}]
        job = optimizer.submit("https://example.com/img.jpg", specs)
        fetched = optimizer.get_job(job.id)
        assert fetched is not None
        assert len(fetched.variants_list) == 1

    def test_get_nonexistent_job(self, optimizer):
        assert optimizer.get_job("nonexistent") is None


class TestProcessing:
    def test_process_job_completes(self, optimizer):
        job = optimizer.submit("https://example.com/photo.jpg", [
            {"name": "sm", "width": 400, "height": 300, "format": "webp", "quality": 85}
        ])
        completed = optimizer.process(job.id)
        assert completed.status == JobStatus.COMPLETED.value

    def test_process_creates_variants(self, optimizer):
        job = optimizer.submit("https://example.com/photo.jpg", [
            {"name": "sm", "width": 400, "height": 300, "format": "webp", "quality": 85},
            {"name": "lg", "width": 1200, "height": 900, "format": "jpeg", "quality": 90},
        ])
        optimizer.process(job.id)
        variants = optimizer.get_variants(job.id)
        assert len(variants) == 2

    def test_process_creates_cdn_entries(self, optimizer):
        job = optimizer.submit("https://example.com/photo.jpg", [
            {"name": "md", "width": 800, "height": 600, "format": "webp", "quality": 85},
        ])
        optimizer.process(job.id)
        variants = optimizer.get_variants(job.id)
        cdn = optimizer._get_cdn_for_variant(variants[0].id)
        assert cdn is not None
        assert "cdn.blackroad.io" in cdn.cdn_url

    def test_process_calculates_savings(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/hero.jpg")
        optimizer.process(job.id)
        fetched = optimizer.get_job(job.id)
        assert fetched.total_output_bytes > 0
        assert fetched.savings_percent >= 0


class TestResponsiveSet:
    def test_create_responsive_set(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/img.jpg")
        assert job is not None
        assert len(job.variants_list) == len(ImageOptimizer.RESPONSIVE_VARIANTS)

    def test_responsive_set_has_mobile(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/img.jpg")
        names = [v["name"] for v in job.variants_list]
        assert "mobile" in names
        assert "desktop" in names
        assert "retina" in names


class TestCDN:
    def test_purge_cdn(self, optimizer):
        job = optimizer.submit("https://example.com/img.jpg", [
            {"name": "sm", "width": 300, "height": 200, "format": "webp", "quality": 80}
        ])
        optimizer.process(job.id)
        variants = optimizer.get_variants(job.id)
        result = optimizer.purge_cdn(variants[0].id)
        assert result is True
        cdn = optimizer._get_cdn_for_variant(variants[0].id)
        assert cdn is None  # purged

    def test_record_cdn_hit(self, optimizer):
        job = optimizer.submit("https://example.com/img.jpg", [
            {"name": "sm", "width": 300, "height": 200, "format": "webp", "quality": 80}
        ])
        optimizer.process(job.id)
        variants = optimizer.get_variants(job.id)
        result = optimizer.record_cdn_hit(variants[0].id)
        assert result is True


class TestMetricsAndSrcset:
    def test_get_metrics(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/img.jpg")
        optimizer.process(job.id)
        metrics = optimizer.get_metrics(job.id)
        assert "variant_count" in metrics
        assert metrics["variant_count"] > 0

    def test_generate_srcset(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/img.jpg")
        optimizer.process(job.id)
        srcset = optimizer.generate_srcset(job.id)
        assert len(srcset) > 0
        assert "w" in srcset

    def test_estimate_savings(self, optimizer):
        result = optimizer.estimate_savings("https://example.com/large.jpg", "avif", 80)
        assert "estimated_savings_percent" in result
        assert result["estimated_source_bytes"] > 0

    def test_format_breakdown(self, optimizer):
        job = optimizer.create_responsive_set("https://example.com/img.jpg")
        optimizer.process(job.id)
        breakdown = optimizer.get_format_breakdown(job.id)
        assert len(breakdown) > 0

    def test_batch_optimize(self, optimizer):
        results = optimizer.optimize_batch([
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
        ])
        assert len(results) == 2
        for r in results:
            assert r["status"] == "completed"

    def test_list_jobs(self, optimizer):
        optimizer.submit("https://example.com/img.jpg", [])
        jobs = optimizer.list_jobs()
        assert len(jobs) >= 1

    def test_cdn_stats(self, optimizer):
        stats = optimizer.get_cdn_stats()
        assert "total_cdn_entries" in stats

    def test_create_optimizer_factory(self, tmp_path):
        opt = create_optimizer(str(tmp_path / "factory.db"))
        assert opt is not None
