"""Setup isolation contracts; no pip install, venv creation, network or raw data."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import prepare_ibvp as prep


class InterpreterContractTests(unittest.TestCase):
    def test_supported_python_312_64bit(self):
        self.assertIsNone(prep.check_setup_python(version=(3, 12, 0), bits=64))
        self.assertIsNone(prep.check_setup_python(version=(3, 12, 99), bits=64))

    def test_unsupported_python_versions_fail_before_install(self):
        for version in [(3, 8, 20), (3, 10, 16), (3, 11, 10), (3, 13, 0), (4, 0, 0)]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                prep.check_setup_python(version=version, bits=64)

    def test_32bit_interpreter_rejected(self):
        with self.assertRaises(ValueError):
            prep.check_setup_python(version=(3, 12, 0), bits=32)

    def test_environment_interpreter_path_preserves_spaces(self):
        directory = Path('directory with spaces') / 'ibvp environment'
        expected = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.assertEqual(Path(prep.env_python(directory)), expected)


class ChildEnvironmentTests(unittest.TestCase):
    def test_inherited_python_and_pip_target_overrides_removed_without_parent_mutation(self):
        polluted = {'PYTHONPATH': 'host packages', 'PYTHONHOME': 'host interpreter',
                    'PIP_TARGET': 'global location', 'PIP_PREFIX': 'host prefix',
                    'PIP_USER': '1', 'PATH': 'system executables', 'IBVP_SENTINEL': 'keep me'}
        with mock.patch.dict(os.environ, polluted, clear=True):
            before = dict(os.environ)
            child = prep.clean_child_env()
            self.assertEqual(dict(os.environ), before)
            for key in ['PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER']:
                self.assertNotIn(key, child)
            self.assertEqual(child['PATH'], polluted['PATH'])
            self.assertEqual(child['IBVP_SENTINEL'], 'keep me')
            child['IBVP_SENTINEL'] = 'changed child'
            self.assertEqual(os.environ['IBVP_SENTINEL'], 'keep me')


class InstallCommandTests(unittest.TestCase):
    def check_commands(self, platform_name):
        python = str(Path('environment with spaces') / 'python.exe')
        commands = prep.setup_pip_commands(python, platform_name=platform_name)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIsInstance(command, list)
            self.assertTrue(all(isinstance(argument, str) for argument in command))
            self.assertEqual(command[0], python)
            self.assertIn('-m', command)
            self.assertIn('pip', command)
            self.assertIn('install', command)
            self.assertIn('--isolated', command)
            self.assertIn('--require-virtualenv', command)
            self.assertNotIn('--user', command)
            self.assertNotIn('--target', command)
            self.assertNotIn('--prefix', command)
            self.assertNotIn('-r', command)
            self.assertNotIn('--requirement', command)
        torch_command, other_command = commands
        self.assertIn('torch==2.6.0', torch_command)
        for requirement in ['numpy==1.26.4', 'opencv-contrib-python==4.11.0.86', 'mediapipe==0.10.32']:
            self.assertIn(requirement, other_command)
        self.assertIn('https://pypi.org/simple', other_command)
        return commands

    def test_windows_torch_uses_official_cpu_index(self):
        commands = self.check_commands('Windows')
        self.assertIn('https://download.pytorch.org/whl/cpu', commands[0])

    def test_linux_torch_uses_official_cpu_index(self):
        commands = self.check_commands('Linux')
        self.assertIn('https://download.pytorch.org/whl/cpu', commands[0])

    def test_macos_torch_uses_pypi(self):
        commands = self.check_commands('Darwin')
        self.assertIn('https://pypi.org/simple', commands[0])
        self.assertNotIn('https://download.pytorch.org/whl/cpu', commands[0])


class TargetOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script_dir = self.root / 'project'
        self.script_dir.mkdir()
        self.script = self.script_dir / 'prepare_ibvp.py'
        self.script.write_text('# stand-in\n', encoding='utf-8')

    def test_new_target_allowed_without_creating_it(self):
        target = self.root / 'new environment'
        prep.validate_env_target(target, script_path=self.script)
        self.assertFalse(target.exists())

    def test_existing_empty_target_allowed_without_modification(self):
        target = self.root / 'empty'
        target.mkdir()
        prep.validate_env_target(target, script_path=self.script)
        self.assertEqual(list(target.iterdir()), [])

    def test_owned_environment_with_exact_marker_allowed(self):
        target = self.root / 'owned'; target.mkdir()
        marker = target / '.ibvp-managed.json'
        marker.write_text(json.dumps({'managed_by': 'prepare_ibvp', 'schema': 1}), encoding='utf-8')
        sentinel = target / 'existing.file'; sentinel.write_bytes(b'keep')
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        prep.validate_env_target(target, script_path=self.script)
        self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, before)

    def test_unmarked_nonempty_target_rejected_without_touching_files(self):
        target = self.root / 'user environment'; target.mkdir()
        sentinel = target / 'user.file'; sentinel.write_bytes(b'valuable existing data')
        with self.assertRaises(ValueError):
            prep.validate_env_target(target, script_path=self.script)
        self.assertEqual(sentinel.read_bytes(), b'valuable existing data')
        self.assertEqual([p.name for p in target.iterdir()], ['user.file'])

    def test_invalid_or_unrelated_marker_rejected(self):
        for index, text in enumerate(['not json', '{}', '[]', 'null', json.dumps({'managed_by': 'other', 'schema': 1}),
                                      json.dumps({'managed_by': 'prepare_ibvp', 'schema': 2})]):
            with self.subTest(marker=text):
                target = self.root / ('bad-marker-' + str(index)); target.mkdir()
                marker = target / '.ibvp-managed.json'; marker.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    prep.validate_env_target(target, script_path=self.script)
                self.assertEqual(marker.read_text(encoding='utf-8'), text)

    def test_regular_file_cannot_be_environment_target(self):
        target = self.root / 'file'; target.write_bytes(b'preserve')
        with self.assertRaises(ValueError):
            prep.validate_env_target(target, script_path=self.script)
        self.assertEqual(target.read_bytes(), b'preserve')

    def test_script_directory_and_ancestors_rejected(self):
        for target in [self.script_dir, self.root, self.root.parent]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                prep.validate_env_target(target, script_path=self.script)

    def test_active_and_base_interpreters_rejected_even_if_owned(self):
        active = self.root / 'active'; base = self.root / 'base'
        for target in [active, base]:
            target.mkdir()
            (target / '.ibvp-managed.json').write_text(json.dumps({'managed_by': 'prepare_ibvp', 'schema': 1}), encoding='utf-8')
        with mock.patch.object(sys, 'prefix', str(active)), mock.patch.object(sys, 'base_prefix', str(base)):
            for target in [active, base]:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    prep.validate_env_target(target, script_path=self.script)


class ArgumentForwardingTests(unittest.TestCase):
    def test_setup_and_environment_option_removed_preserving_paths_and_order(self):
        original = ['--setup', '--input-dir', 'raw data with spaces', '--env-dir', 'env with spaces',
                    '--output-dir', 'processed data', '--sessions', 'p32_d', 'p22_a']
        expected = ['--input-dir', 'raw data with spaces', '--output-dir', 'processed data',
                    '--sessions', 'p32_d', 'p22_a']
        before = original.copy()
        self.assertEqual(prep.strip_setup_args(original), expected)
        self.assertEqual(original, before)

    def test_equals_environment_option_removed(self):
        self.assertEqual(prep.strip_setup_args(['--inspect', '--env-dir=env with spaces', '--setup']), ['--inspect'])

    def test_unrelated_similar_options_and_values_preserved(self):
        original = ['--input-dir', 'a--setup-folder', '--output-dir=env-dir outputs', '--seed', '123']
        self.assertEqual(prep.strip_setup_args(original), original)

    def test_multiple_setup_options_cannot_recur_on_relaunch(self):
        args = ['--setup', '--env-dir', 'one', '--setup', '--env-dir=two', '--inspect']
        self.assertEqual(prep.strip_setup_args(args), ['--inspect'])


class BootstrapRoutingTests(unittest.TestCase):
    def test_existing_setup_forwards_paths_and_canonical_env_without_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'environment with spaces'
            target.mkdir()
            (target / '.ibvp-managed.json').write_text(json.dumps({
                'managed_by': 'prepare_ibvp', 'schema': 1, 'status': 'ready',
                'recipe': prep.SETUP_RECIPE}), encoding='utf-8')
            executable = prep.env_python(target)
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b'not executed')
            args = ['--env-dir', str(target), '--input-dir', 'raw data',
                    '--output-dir', 'processed data', '--sessions', 'p22_a']
            with mock.patch.object(prep, 'install_setup') as install, \
                 mock.patch.object(prep.subprocess, 'run', return_value=mock.Mock(returncode=7)) as run:
                self.assertEqual(prep.standalone_bootstrap(args), 7)
            install.assert_not_called()
            command = run.call_args.args[0]
            self.assertEqual(command[:3], [str(executable.resolve()), '-I', str(Path(prep.__file__).resolve())])
            self.assertEqual(command[3:], ['--input-dir', 'raw data', '--output-dir', 'processed data',
                                          '--sessions', 'p22_a', '--env-dir', str(target.resolve())])

    def test_setup_current_env_conflict_rejected_before_install(self):
        with mock.patch.object(prep, 'install_setup') as install, self.assertRaises(ValueError):
            prep.standalone_bootstrap(['--setup', '--use-current-env'])
        install.assert_not_called()

    def test_nondict_managed_marker_rejected_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'environment'
            target.mkdir()
            (target / '.ibvp-managed.json').write_text('[]', encoding='utf-8')
            with mock.patch.object(prep.subprocess, 'run') as run, self.assertRaises(ValueError):
                prep.standalone_bootstrap(['--env-dir', str(target), '--inspect'])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
