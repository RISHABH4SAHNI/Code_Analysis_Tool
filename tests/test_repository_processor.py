import textwrap
from core.repository_processor import RepositoryProcessor


def write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def by_name(functions):
    return {f"{f.class_context}.{f.name}" if f.class_context else f.name: f for f in functions}


def test_class_context_and_self_calls(tmp_path):
    write(tmp_path / "app" / "service.py", """
        import os

        class Repo:
            def save(self):
                return self.validate()

            def validate(self):
                return True

        class Cache:
            def validate(self):
                return False

        def run():
            return helper()

        def helper():
            return os.getcwd()
    """)
    functions = by_name(RepositoryProcessor().process_repository(str(tmp_path)))

    assert set(functions) == {"Repo.save", "Repo.validate", "Cache.validate", "run", "helper"}
    # A module-level function after a class is not a method of that class
    assert functions["run"].class_context is None
    # self.validate() resolves to the caller's own class, not every "validate"
    assert functions["Repo.save"].nested_call_ids == [functions["Repo.validate"].lookup_id]
    assert functions["run"].nested_call_ids == [functions["helper"].lookup_id]
    assert functions["helper"].imports == ["os"]


def test_lookup_ids_are_stable_across_runs(tmp_path):
    write(tmp_path / "mod.py", "def a():\n    pass\n")
    first = RepositoryProcessor().process_repository(str(tmp_path))
    second = RepositoryProcessor().process_repository(str(tmp_path))
    assert [f.lookup_id for f in first] == [f.lookup_id for f in second]


def test_tests_and_virtualenvs_are_skipped_by_default(tmp_path):
    write(tmp_path / "src" / "core.py", "def core():\n    pass\n")
    write(tmp_path / "tests" / "test_core.py", "def test_core():\n    pass\n")
    write(tmp_path / ".venv" / "lib" / "dep.py", "def dep():\n    pass\n")

    names = {f.name for f in RepositoryProcessor().process_repository(str(tmp_path))}
    assert names == {"core"}

    names = {f.name for f in RepositoryProcessor().process_repository(str(tmp_path), include_tests=True)}
    assert names == {"core", "test_core"}


def test_unparseable_files_are_reported(tmp_path):
    write(tmp_path / "good.py", "def good():\n    pass\n")
    write(tmp_path / "bad.py", "def broken(:\n")
    processor = RepositoryProcessor()

    functions = processor.process_repository(str(tmp_path))

    assert [f.name for f in functions] == ["good"]
    assert len(processor.skipped_files) == 1
    assert processor.skipped_files[0]["file"].endswith("bad.py")


def test_decorated_function_code_matches_line_numbers(tmp_path):
    write(tmp_path / "views.py", """
        def cached(fn):
            return fn

        @cached
        def view():
            try:
                return 1
            except (KeyError, ValueError):
                return 0
    """)
    view = by_name(RepositoryProcessor().process_repository(str(tmp_path)))["view"]

    assert view.code.splitlines()[0] == "@cached"
    assert view.code_with_line_numbers.splitlines()[0].strip() == f"{view.start_line}: @cached"
    assert view.error_handling["error_types"] == ["KeyError", "ValueError"]
