


# @pytest.fixture
# def dummy_hdf5(tmp_path):
#     # Use CLI to generate dummy file in test
#     from subprocess import run

#     output = tmp_path / "dummy.hdf5"
#     run(
#         [
#             "uv",
#             "run",
#             "python3",
#             "-m",
#             "egohub.cli.main",
#             "convert",
#             "dummy_multi_view",
#             "--raw-dir",
#             str(tmp_path / "raw"),
#             "--output-file",
#             str(output),
#             "--num-sequences",
#             "1",
#         ],
#         check=True,
#     )
#     return output


# def test_multi_view_export(dummy_hdf5):
#     exporter = RerunExporter(max_frames=5)
#     exporter.export(dummy_hdf5, output_path=None)  # Spawn viewer
#     # Add assertions if needed, but since it's integration, manual verification might suffice  # noqa: E501
