"""
Unit tests for migration manager functionality.

Tests comprehensive migration capabilities, plan generation,
execution, and rollback functionality.
"""

import pytest
import tempfile
from pathlib import Path

from src.versioning.artifact_manager import ArtifactManager
from src.versioning.migration_manager import (
    MigrationManager,
    MigrationStep,
    MigrationPlan,
    MigrationResult,
    MigrationError,
    create_builtin_migration_steps,
)


class TestMigrationStep:
    """Test MigrationStep functionality."""

    def test_migration_step_creation(self):
        """Test creating migration step."""

        def migration_func(context):
            pass

        def rollback_func(context):
            pass

        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=migration_func,
            rollback_function=rollback_func,
            dependencies=["step1", "step2"],
            required_artifacts=["model.pt"],
            created_artifacts=["model_v2.pt"],
        )

        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.from_version == "v1.0.0"
        assert step.to_version == "v2.0.0"
        assert step.migration_function == migration_func
        assert step.rollback_function == rollback_func
        assert step.dependencies == ["step1", "step2"]
        assert step.required_artifacts == ["model.pt"]
        assert step.created_artifacts == ["model_v2.pt"]

    def test_migration_step_to_dict(self):
        """Test converting migration step to dictionary."""

        def migration_func(context):
            pass

        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=migration_func,
        )

        data = step.to_dict()

        assert data["step_id"] == "test_step"
        assert data["name"] == "Test Step"
        assert data["from_version"] == "v1.0.0"
        assert data["to_version"] == "v2.0.0"
        assert "migration_function" not in data  # Functions not serialized


class TestMigrationPlan:
    """Test MigrationPlan functionality."""

    def test_migration_plan_creation(self):
        """Test creating migration plan."""
        step1 = MigrationStep(
            step_id="step1",
            name="Step 1",
            description="First step",
            from_version="v1.0.0",
            to_version="v1.5.0",
            migration_function=lambda x: None,
        )

        step2 = MigrationStep(
            step_id="step2",
            name="Step 2",
            description="Second step",
            from_version="v1.5.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        plan = MigrationPlan(
            plan_id="test_plan",
            from_version="v1.0.0",
            to_version="v2.0.0",
            steps=[step1, step2],
            created_at="2024-01-01T00:00:00Z",
            description="Test migration plan",
        )

        assert plan.plan_id == "test_plan"
        assert plan.from_version == "v1.0.0"
        assert plan.to_version == "v2.0.0"
        assert len(plan.steps) == 2
        assert plan.description == "Test migration plan"

    def test_migration_plan_to_dict(self):
        """Test converting migration plan to dictionary."""
        step = MigrationStep(
            step_id="step1",
            name="Step 1",
            description="Test step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        plan = MigrationPlan(
            plan_id="test_plan",
            from_version="v1.0.0",
            to_version="v2.0.0",
            steps=[step],
            created_at="2024-01-01T00:00:00Z",
        )

        data = plan.to_dict()

        assert data["plan_id"] == "test_plan"
        assert data["from_version"] == "v1.0.0"
        assert data["to_version"] == "v2.0.0"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["step_id"] == "step1"


class TestMigrationResult:
    """Test MigrationResult functionality."""

    def test_migration_result_creation(self):
        """Test creating migration result."""
        result = MigrationResult(
            migration_id="migration_123",
            plan_id="plan_456",
            from_version="v1.0.0",
            to_version="v2.0.0",
            status="completed",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z",
            steps_completed=["step1", "step2"],
            steps_failed=[],
            backup_location="/backup/path",
        )

        assert result.migration_id == "migration_123"
        assert result.plan_id == "plan_456"
        assert result.status == "completed"
        assert len(result.steps_completed) == 2
        assert len(result.steps_failed) == 0
        assert result.backup_location == "/backup/path"

    def test_migration_result_to_dict(self):
        """Test converting migration result to dictionary."""
        result = MigrationResult(
            migration_id="migration_123",
            plan_id="plan_456",
            from_version="v1.0.0",
            to_version="v2.0.0",
            status="completed",
            started_at="2024-01-01T00:00:00Z",
        )

        data = result.to_dict()

        assert data["migration_id"] == "migration_123"
        assert data["plan_id"] == "plan_456"
        assert data["status"] == "completed"
        assert data["from_version"] == "v1.0.0"
        assert data["to_version"] == "v2.0.0"


class TestMigrationManager:
    """Test MigrationManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def artifact_manager(self, temp_dir):
        """Create ArtifactManager instance for testing."""
        return ArtifactManager(temp_dir / "artifacts")

    @pytest.fixture
    def migration_manager(self, artifact_manager):
        """Create MigrationManager instance for testing."""
        return MigrationManager(artifact_manager)

    def test_initialization(self, artifact_manager):
        """Test migration manager initialization."""
        manager = MigrationManager(artifact_manager)

        assert manager.artifact_manager == artifact_manager
        assert len(manager.migration_steps) == 0
        assert len(manager.migration_plans) == 0
        assert len(manager.migration_results) == 0
        assert manager.migrations_dir.exists()

    def test_register_migration_step(self, migration_manager):
        """Test registering migration step."""

        def migration_func(context):
            pass

        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=migration_func,
        )

        migration_manager.register_migration_step(step)

        assert "test_step" in migration_manager.migration_steps
        assert migration_manager.migration_steps["test_step"] == step

    def test_create_migration_plan(self, migration_manager):
        """Test creating migration plan."""
        # Register migration steps
        step1 = MigrationStep(
            step_id="step1",
            name="Step 1",
            description="First step",
            from_version="v1.0.0",
            to_version="v1.5.0",
            migration_function=lambda x: None,
        )

        step2 = MigrationStep(
            step_id="step2",
            name="Step 2",
            description="Second step",
            from_version="v1.5.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        migration_manager.register_migration_step(step1)
        migration_manager.register_migration_step(step2)

        # Create migration plan
        plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0", "Test plan")

        assert plan.from_version == "v1.0.0"
        assert plan.to_version == "v2.0.0"
        assert plan.description == "Test plan"
        assert len(plan.steps) == 2
        assert plan.plan_id in migration_manager.migration_plans

    def test_create_migration_plan_invalid_versions(self, migration_manager):
        """Test creating migration plan with invalid versions."""
        with pytest.raises(MigrationError) as exc_info:
            migration_manager.create_migration_plan("v2.0.0", "v1.0.0")

        assert "Target version must be greater" in str(exc_info.value)

    def test_create_migration_plan_no_path(self, migration_manager):
        """Test creating migration plan with no available path."""
        with pytest.raises(MigrationError) as exc_info:
            migration_manager.create_migration_plan("v1.0.0", "v2.0.0")

        assert "No migration path found" in str(exc_info.value)

    def test_execute_migration(self, migration_manager):
        """Test executing migration."""
        # Register migration step
        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=lambda context: None,
        )

        migration_manager.register_migration_step(step)

        # Create migration plan
        plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0")

        # Execute migration
        result = migration_manager.execute_migration(plan.plan_id)

        assert result.plan_id == plan.plan_id
        assert result.status == "completed"
        assert "test_step" in result.steps_completed

    def test_execute_migration_nonexistent_plan(self, migration_manager):
        """Test executing migration with nonexistent plan."""
        with pytest.raises(MigrationError) as exc_info:
            migration_manager.execute_migration("nonexistent_plan")

        assert "Migration plan not found" in str(exc_info.value)

    def test_rollback_migration(self, migration_manager):
        """Test rolling back migration."""

        # Register migration step with rollback
        def migration_func(context):
            pass

        def rollback_func(context):
            pass

        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=migration_func,
            rollback_function=rollback_func,
        )

        migration_manager.register_migration_step(step)

        # Create and execute migration
        plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0")
        result = migration_manager.execute_migration(plan.plan_id)

        # Rollback migration
        rollback_result = migration_manager.rollback_migration(result.migration_id)

        assert rollback_result.from_version == "v2.0.0"
        assert rollback_result.to_version == "v1.0.0"
        assert rollback_result.status == "completed"
        assert f"rollback_{step.step_id}" in rollback_result.steps_completed

    def test_rollback_migration_nonexistent(self, migration_manager):
        """Test rolling back nonexistent migration."""
        with pytest.raises(MigrationError) as exc_info:
            migration_manager.rollback_migration("nonexistent_migration")

        assert "Migration result not found" in str(exc_info.value)

    def test_rollback_migration_invalid_status(self, migration_manager):
        """Test rolling back migration with invalid status."""
        # Create a failed migration result
        result = MigrationResult(
            migration_id="test_migration",
            plan_id="test_plan",
            from_version="v1.0.0",
            to_version="v2.0.0",
            status="failed",
            started_at="2024-01-01T00:00:00Z",
        )

        migration_manager.migration_results["test_migration"] = result

        with pytest.raises(MigrationError) as exc_info:
            migration_manager.rollback_migration("test_migration")

        assert "Cannot rollback migration with status" in str(exc_info.value)

    def test_list_migration_plans(self, migration_manager):
        """Test listing migration plans."""
        # Register steps and create plans
        step1 = MigrationStep(
            step_id="test_step1",
            name="Test Step 1",
            description="Test migration step 1",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        step2 = MigrationStep(
            step_id="test_step2",
            name="Test Step 2",
            description="Test migration step 2",
            from_version="v2.0.0",
            to_version="v3.0.0",
            migration_function=lambda x: None,
        )

        migration_manager.register_migration_step(step1)
        migration_manager.register_migration_step(step2)

        plan1 = migration_manager.create_migration_plan("v1.0.0", "v2.0.0", "Plan 1")
        plan2 = migration_manager.create_migration_plan("v2.0.0", "v3.0.0", "Plan 2")

        plans = migration_manager.list_migration_plans()

        assert len(plans) == 2
        assert plan1 in plans
        assert plan2 in plans

    def test_list_migration_results(self, migration_manager):
        """Test listing migration results."""
        # Create migration results
        result1 = MigrationResult(
            migration_id="migration_1",
            plan_id="plan_1",
            from_version="v1.0.0",
            to_version="v2.0.0",
            status="completed",
            started_at="2024-01-01T00:00:00Z",
        )

        result2 = MigrationResult(
            migration_id="migration_2",
            plan_id="plan_2",
            from_version="v2.0.0",
            to_version="v3.0.0",
            status="failed",
            started_at="2024-01-01T00:00:00Z",
        )

        migration_manager.migration_results["migration_1"] = result1
        migration_manager.migration_results["migration_2"] = result2

        # List all results
        all_results = migration_manager.list_migration_results()
        assert len(all_results) == 2

        # List completed results
        completed_results = migration_manager.list_migration_results("completed")
        assert len(completed_results) == 1
        assert completed_results[0].migration_id == "migration_1"

        # List failed results
        failed_results = migration_manager.list_migration_results("failed")
        assert len(failed_results) == 1
        assert failed_results[0].migration_id == "migration_2"

    def test_get_migration_plan(self, migration_manager):
        """Test getting migration plan."""
        # Register step and create plan
        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        migration_manager.register_migration_step(step)
        plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0")

        # Get plan
        retrieved_plan = migration_manager.get_migration_plan(plan.plan_id)
        assert retrieved_plan == plan

        # Get nonexistent plan
        nonexistent_plan = migration_manager.get_migration_plan("nonexistent")
        assert nonexistent_plan is None

    def test_get_migration_result(self, migration_manager):
        """Test getting migration result."""
        # Create migration result
        result = MigrationResult(
            migration_id="test_migration",
            plan_id="test_plan",
            from_version="v1.0.0",
            to_version="v2.0.0",
            status="completed",
            started_at="2024-01-01T00:00:00Z",
        )

        migration_manager.migration_results["test_migration"] = result

        # Get result
        retrieved_result = migration_manager.get_migration_result("test_migration")
        assert retrieved_result == result

        # Get nonexistent result
        nonexistent_result = migration_manager.get_migration_result("nonexistent")
        assert nonexistent_result is None

    def test_validate_migration_plan(self, migration_manager):
        """Test validating migration plan."""
        # Register step
        step = MigrationStep(
            step_id="test_step",
            name="Test Step",
            description="Test migration step",
            from_version="v1.0.0",
            to_version="v2.0.0",
            migration_function=lambda x: None,
        )

        migration_manager.register_migration_step(step)

        # Create plan
        plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0")

        # Validate plan
        assert migration_manager.validate_migration_plan(plan.plan_id) is True

        # Validate nonexistent plan
        assert migration_manager.validate_migration_plan("nonexistent") is False


class TestBuiltinMigrationSteps:
    """Test built-in migration steps."""

    def test_create_builtin_migration_steps(self):
        """Test creating built-in migration steps."""
        steps = create_builtin_migration_steps()

        assert len(steps) == 2

        step_ids = [step.step_id for step in steps]
        assert "schema_upgrade_v1_v2" in step_ids
        assert "data_migration_v1_v2" in step_ids

        # Check schema upgrade step
        schema_step = next(s for s in steps if s.step_id == "schema_upgrade_v1_v2")
        assert schema_step.from_version == "v1.0.0"
        assert schema_step.to_version == "v2.0.0"
        assert "model.pt" in schema_step.required_artifacts
        assert "config.yaml" in schema_step.required_artifacts
        assert "schema_v2.json" in schema_step.created_artifacts

        # Check data migration step
        data_step = next(s for s in steps if s.step_id == "data_migration_v1_v2")
        assert data_step.from_version == "v1.0.0"
        assert data_step.to_version == "v2.0.0"
        assert "schema_upgrade_v1_v2" in data_step.dependencies
        assert "model.pt" in data_step.required_artifacts
        assert "model_v2.pt" in data_step.created_artifacts


class TestMigrationManagerIntegration:
    """Integration tests for migration manager."""

    def test_end_to_end_migration_workflow(self):
        """Test end-to-end migration workflow."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up artifact manager
            artifact_manager = ArtifactManager(Path(temp_dir) / "artifacts")
            migration_manager = MigrationManager(artifact_manager)

            # Register built-in steps
            for step in create_builtin_migration_steps():
                migration_manager.register_migration_step(step)

            # Create migration plan
            plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0", "Test migration")

            assert plan.from_version == "v1.0.0"
            assert plan.to_version == "v2.0.0"
            assert len(plan.steps) > 0

            # Validate plan
            assert migration_manager.validate_migration_plan(plan.plan_id) is True

            # Execute migration (validation is skipped in test environment)
            result = migration_manager.execute_migration(plan.plan_id)
            assert result.status == "completed"

    def test_migration_with_rollback(self):
        """Test migration with rollback capability."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up artifact manager
            artifact_manager = ArtifactManager(Path(temp_dir) / "artifacts")
            migration_manager = MigrationManager(artifact_manager)

            # Create custom migration step with rollback
            def migration_func(context):
                context["result"].metadata["migrated"] = True

            def rollback_func(context):
                context["result"].metadata["rolled_back"] = True

            step = MigrationStep(
                step_id="test_migration",
                name="Test Migration",
                description="Test migration with rollback",
                from_version="v1.0.0",
                to_version="v2.0.0",
                migration_function=migration_func,
                rollback_function=rollback_func,
            )

            migration_manager.register_migration_step(step)

            # Create and execute migration
            plan = migration_manager.create_migration_plan("v1.0.0", "v2.0.0")
            result = migration_manager.execute_migration(plan.plan_id)

            assert result.status == "completed"
            assert result.metadata.get("migrated") is True

            # Rollback migration
            rollback_result = migration_manager.rollback_migration(result.migration_id)

            assert rollback_result.status == "completed"
            assert rollback_result.metadata.get("rolled_back") is True
