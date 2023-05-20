from contrasttransferfunction.utils import calculate_diagonal_radius


def test_diagonal_radius():
    assert calculate_diagonal_radius(512) == 361
