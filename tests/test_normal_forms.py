import pathlib
import pytest
import numpy as np

from xnlbd.analyse.normal_forms import *

test_data_folder = pathlib.Path(__file__).parent.joinpath("test_data").absolute()


def test_comparison_to_fortran_nonresonant():
    test_map = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/map_nonres.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            test_map.x_poly.terms.extend(new_terms)
        elif j == 1:
            test_map.px_poly.terms.extend(new_terms)
        elif j == 2:
            test_map.y_poly.terms.extend(new_terms)
        else:
            test_map.py_poly.terms.extend(new_terms)
    f.close()

    test_nf = NormalForm4D(
        complex_map=test_map,
        max_map_order=map_order,
        max_nf_order=10,
        res_space_dim=0,
        res_case=0,
    )

    test_nf.compute_normal_form()

    ground_truth_Phi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/phi_nonres.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Phi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Phi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Phi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Phi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Phi.x_poly.remove_zero_terms()
    ground_truth_Phi.px_poly.remove_zero_terms()
    ground_truth_Phi.y_poly.remove_zero_terms()
    ground_truth_Phi.py_poly.remove_zero_terms()

    ground_truth_Psi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/psi_nonres.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Psi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Psi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Psi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Psi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Psi.x_poly.remove_zero_terms()
    ground_truth_Psi.px_poly.remove_zero_terms()
    ground_truth_Psi.y_poly.remove_zero_terms()
    ground_truth_Psi.py_poly.remove_zero_terms()

    ground_truth_H = Polynom(terms=[])
    f = open(test_data_folder.joinpath("normal_forms/hamil_nonres.dat"))
    f_lines = f.readlines()
    new_terms = []
    for iline in range(1, len(f_lines)):
        line = f_lines[iline]
        new_terms.append(
            Term(
                coeff=complex(float(line.split()[0]), float(line.split()[1])),
                x_exp=int(line.split()[2]),
                px_exp=int(line.split()[3]),
                y_exp=int(line.split()[4]),
                py_exp=int(line.split()[5]),
            )
        )
    ground_truth_H.terms = new_terms
    ground_truth_H.remove_zero_terms()
    f.close()

    assert test_nf.Phi == ground_truth_Phi
    assert test_nf.Psi == ground_truth_Psi
    assert test_nf.H == ground_truth_H


def test_comparison_to_fortran_resonant():
    test_map = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/map_res.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            test_map.x_poly.terms.extend(new_terms)
        elif j == 1:
            test_map.px_poly.terms.extend(new_terms)
        elif j == 2:
            test_map.y_poly.terms.extend(new_terms)
        else:
            test_map.py_poly.terms.extend(new_terms)
    f.close()

    test_nf = NormalForm4D(
        complex_map=test_map,
        max_map_order=map_order,
        max_nf_order=10,
        res_space_dim=1,
        res_case=1,
        res_eig=[0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        res_basis1=[2, 1],
    )

    test_nf.compute_normal_form()

    ground_truth_Phi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/phi_res.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Phi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Phi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Phi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Phi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Phi.x_poly.remove_zero_terms()
    ground_truth_Phi.px_poly.remove_zero_terms()
    ground_truth_Phi.y_poly.remove_zero_terms()
    ground_truth_Phi.py_poly.remove_zero_terms()

    ground_truth_Psi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/psi_res.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Psi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Psi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Psi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Psi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Psi.x_poly.remove_zero_terms()
    ground_truth_Psi.px_poly.remove_zero_terms()
    ground_truth_Psi.y_poly.remove_zero_terms()
    ground_truth_Psi.py_poly.remove_zero_terms()

    ground_truth_H = Polynom(terms=[])
    f = open(test_data_folder.joinpath("normal_forms/hamil_res.dat"))
    f_lines = f.readlines()
    new_terms = []
    for iline in range(1, len(f_lines)):
        line = f_lines[iline]
        new_terms.append(
            Term(
                coeff=complex(float(line.split()[0]), float(line.split()[1])),
                x_exp=int(line.split()[2]),
                px_exp=int(line.split()[3]),
                y_exp=int(line.split()[4]),
                py_exp=int(line.split()[5]),
            )
        )
    ground_truth_H.terms = new_terms
    ground_truth_H.remove_zero_terms()
    f.close()

    assert test_nf.Phi == ground_truth_Phi
    assert test_nf.Psi == ground_truth_Psi
    assert test_nf.H == ground_truth_H


def test_comparison_to_fortran_quasiresonant():
    test_map = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/map_quasires.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            test_map.x_poly.terms.extend(new_terms)
        elif j == 1:
            test_map.px_poly.terms.extend(new_terms)
        elif j == 2:
            test_map.y_poly.terms.extend(new_terms)
        else:
            test_map.py_poly.terms.extend(new_terms)
    f.close()

    test_nf = NormalForm4D(
        complex_map=test_map,
        max_map_order=map_order,
        max_nf_order=10,
        res_space_dim=1,
        res_case=2,
        res_eig=[
            complex(0.9776120391412251, 0.2104155434518462),
            complex(0.9776120391412251, -0.2104155434518462),
            complex(0.9943872583508965, 0.1058016088223022),
            complex(0.9943872583508965, -0.1058016088223022),
        ],
        res_basis1=[1, -2],
    )

    test_nf.compute_normal_form()

    ground_truth_Phi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/phi_quasires.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Phi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Phi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Phi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Phi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Phi.x_poly.remove_zero_terms()
    ground_truth_Phi.px_poly.remove_zero_terms()
    ground_truth_Phi.y_poly.remove_zero_terms()
    ground_truth_Phi.py_poly.remove_zero_terms()

    ground_truth_Psi = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/psi_quasires.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            ground_truth_Psi.x_poly.terms.extend(new_terms)
        elif j == 1:
            ground_truth_Psi.px_poly.terms.extend(new_terms)
        elif j == 2:
            ground_truth_Psi.y_poly.terms.extend(new_terms)
        else:
            ground_truth_Psi.py_poly.terms.extend(new_terms)
    f.close()
    ground_truth_Psi.x_poly.remove_zero_terms()
    ground_truth_Psi.px_poly.remove_zero_terms()
    ground_truth_Psi.y_poly.remove_zero_terms()
    ground_truth_Psi.py_poly.remove_zero_terms()

    ground_truth_H = Polynom(terms=[])
    f = open(test_data_folder.joinpath("normal_forms/hamil_quasires.dat"))
    f_lines = f.readlines()
    new_terms = []
    for iline in range(1, len(f_lines)):
        line = f_lines[iline]
        new_terms.append(
            Term(
                coeff=complex(float(line.split()[0]), float(line.split()[1])),
                x_exp=int(line.split()[2]),
                px_exp=int(line.split()[3]),
                y_exp=int(line.split()[4]),
                py_exp=int(line.split()[5]),
            )
        )
    ground_truth_H.terms = new_terms
    ground_truth_H.remove_zero_terms()
    f.close()

    max_order = 5
    test_nf.Phi.x_poly.truncate_at_order(max_order)
    test_nf.Psi.x_poly.truncate_at_order(max_order)
    test_nf.H.truncate_at_order(max_order)
    ground_truth_Phi.x_poly.truncate_at_order(max_order)
    ground_truth_Psi.x_poly.truncate_at_order(max_order)
    ground_truth_H.truncate_at_order(max_order)
    print("NF")
    print(test_nf.H)
    print("Truth")
    print(ground_truth_H)

    assert (test_nf.Phi == ground_truth_Phi)
    assert (test_nf.Psi == ground_truth_Psi)
    assert (test_nf.H == ground_truth_H)


def test_argument_edge_cases():
    
    test_map = Map(
        x_poly=Polynom(terms=[]),
        px_poly=Polynom(terms=[]),
        y_poly=Polynom(terms=[]),
        py_poly=Polynom(terms=[]),
    )
    f = open(test_data_folder.joinpath("normal_forms/map_res.dat"))
    f_lines = f.readlines()
    map_order = int(f_lines[0])
    ncoef = int(
        (map_order + 4) * (map_order + 3) * (map_order + 2) * (map_order + 1) / 24
    )
    # Create Map
    iline = 1
    for j in range(0, 4):
        new_terms = []
        for i in range(0, ncoef):
            line = f_lines[iline]
            iline += 1
            new_terms.append(
                Term(
                    coeff=complex(float(line.split()[0]), float(line.split()[1])),
                    x_exp=int(line.split()[2]),
                    px_exp=int(line.split()[3]),
                    y_exp=int(line.split()[4]),
                    py_exp=int(line.split()[5]),
                )
            )
        if j == 0:
            test_map.x_poly.terms.extend(new_terms)
        elif j == 1:
            test_map.px_poly.terms.extend(new_terms)
        elif j == 2:
            test_map.y_poly.terms.extend(new_terms)
        else:
            test_map.py_poly.terms.extend(new_terms)
    f.close()
    with pytest.raises(ValueError) as excinfo:
        # Test need for 2 resonant bases in case of
        # res_space_dim = 2
        test_nf = NormalForm4D(
        complex_map=test_map,
        max_map_order=map_order,
        max_nf_order=10,
        res_space_dim=2,
        res_case=1,
        res_eig=[0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        res_basis1=[2, 1],
        )
        assert "resonant bases" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        # Test whether error is raised if incorrect
        # number of resonant eigenvalues is provided
        test_nf = NormalForm4D(
        complex_map=test_map,
        max_map_order=map_order,
        max_nf_order=10,
        res_space_dim=1,
        res_case=1,
        res_eig=[0 + 0j, 0 + 0j, 0 + 0j],
        res_basis1=[2, 1],
        )
        assert "eigenvalues" in str(excinfo.value)
test_argument_edge_cases()