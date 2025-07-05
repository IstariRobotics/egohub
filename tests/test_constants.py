from egohub import constants


def test_canonical_skeleton_constants(data_regression):
    """
    Creates a snapshot of the canonical skeleton definitions.
    This test will fail if the joint list or hierarchy changes,
    preventing accidental modifications to this core data structure.
    """
    skeleton_data = {
        "joints": constants.CANONICAL_SKELETON_JOINTS,
        "hierarchy": constants.CANONICAL_SKELETON_HIERARCHY,
        "avp_links": constants.AVP_LINKS,
    }
    data_regression.check(skeleton_data) 