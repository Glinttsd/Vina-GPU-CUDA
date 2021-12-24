/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"


#include "random.h"
#include <iostream>

#include <fstream>
#include <boost/progress.hpp>
#include <thread>

#include "common_macro.h"
#include "kernel2.cuh"

//#define DISPLAY_ANALYSIS
//#define DATA_DISTRIBUTION_TEST
//#define BUILD_KERNEL_FROM_SOURCE
#define OPENCL_PART_2
output_type monte_carlo::operator()(model& m, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	output_container tmp;
	this->operator()(m, tmp, p, ig, p_widened, ig_widened, corner1, corner2, increment_me, generator); // call the version that produces the whole container
	VINA_CHECK(!tmp.empty());
	return tmp.front();
}

bool metropolis_accept(fl old_f, fl new_f, fl temperature, rng& generator) {
	if(new_f < old_f) return true;
	const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
	return random_fl(0, 1, generator) < acceptance_probability;
}

void monte_carlo::single_run(model& m, output_type& out, const precalculate& p, const igrid& ig, rng& generator) const {
	conf_size s = m.get_size();
	change g(s);
	vec authentic_v(1000, 1000, 1000);
	out.e = max_fl;
	output_type current(out);
	quasi_newton quasi_newton_par; quasi_newton_par.max_steps = ssd_par.evals;
	VINA_U_FOR(step, num_steps) {
		output_type candidate(current.c, max_fl);
		mutate_conf(candidate.c, m, mutation_amplitude, generator);
		quasi_newton_par(m, p, ig, candidate, g, hunt_cap);
		if(step == 0 || metropolis_accept(current.e, candidate.e, temperature, generator)) {
			quasi_newton_par(m, p, ig, candidate, g, authentic_v);
			current = candidate;
			if(current.e < out.e)
				out = current;
		}
	}
	quasi_newton_par(m, p, ig, out, g, authentic_v);
}

void monte_carlo::many_runs(model& m, output_container& out, const precalculate& p, const igrid& ig, const vec& corner1, const vec& corner2, sz num_runs, rng& generator) const {
	conf_size s = m.get_size();
	VINA_FOR(run, num_runs) {
		output_type tmp(s, 0);
		tmp.c.randomize(corner1, corner2, generator);
		single_run(m, tmp, p, ig, generator);
		out.push_back(new output_type(tmp));
	}
	out.sort();
}

std::vector<output_type> monte_carlo::cl_to_vina(output_type_cl result_ptr[], int exhaus) const {
	std::vector<output_type> results_vina;
	for (int i = 0; i < exhaus; i++) {
		output_type_cl tmp_cl = result_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)tmp_c.ligands[0].rigid.position[j] = tmp_cl.position[j];
		// Orientation
		qt q(tmp_cl.orientation[0], tmp_cl.orientation[1], tmp_cl.orientation[2], tmp_cl.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp_cl.e);
		// torsion
		for (int j = 0; j < tmp_cl.lig_torsion_size; j++)tmp_vina.c.ligands[0].torsions.push_back(tmp_cl.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
			vec v_tmp(tmp_cl.coords[j][0], tmp_cl.coords[j][1], tmp_cl.coords[j][2]);
			if (v_tmp[0] * v_tmp[1] * v_tmp[2] != 0) tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
	}
	return results_vina;
}

void monte_carlo::generate_uniform_position(const vec corner1, const vec corner2, std::vector<vec>& uniform_data, int exhaustiveness) const{
	assert(exhaustiveness == 125);
	int counter = 0;
	int n = 5;
	for (int i = 0; i < n; i++) {
		double p0 = (corner1.data[0] - corner2.data[0]) / n * i + corner2.data[0];
		for (int j = 0; j < n; j++) {
			double p1 = (corner1.data[1] - corner2.data[1]) / n * j + corner2.data[1];
			for (int k = 0; k < n; k++) {
				double p2 = (corner1.data[2] - corner2.data[2]) / n * k + corner2.data[2];
				uniform_data[counter] = vec(p0, p1, p2);
				counter++;
			}
		}
	}
	assert(counter == exhaustiveness);
}

#ifndef OPENCL_PART_2
// out is sorted
void monte_carlo::operator()(model& m, output_container& out, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	tmp.c.randomize(corner1, corner2, generator);
	fl best_e = max_fl;
	quasi_newton quasi_newton_par; quasi_newton_par.max_steps = ssd_par.evals;
	output_type origin = tmp;

	VINA_U_FOR(step, num_steps) {
		if (increment_me)
			++(*increment_me);
		output_type candidate = tmp;
		mutate_conf(candidate.c, m, mutation_amplitude, generator);
		quasi_newton_par(m, p, ig, candidate, g, hunt_cap);
		if (step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, generator)) {
			tmp = candidate;

			m.set(tmp.c); // FIXME? useless?

			// FIXME only for very promising ones
			if (tmp.e < best_e || out.size() < num_saved_mins) {
				quasi_newton_par(m, p, ig, tmp, g, authentic_v);
				m.set(tmp.c); // FIXME? useless?
				tmp.coords = m.get_heavy_atom_movable_coords();
				add_to_output_container(out, tmp, min_rmsd, num_saved_mins); // 20 - max size
				if (tmp.e < best_e)
					best_e = tmp.e;
			}
		}
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order
}
#else

volatile bool finished = false; //20211119 Glinttsd
void print_process() {
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		sleep(0.1);
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)printf(" ");
		printf("|"); fflush(stdout);

		count++;
		count %= 30;
	} while (finished != true);
	printf("\rPerform docking|");
	for (int i = 0; i < 16; i++)printf("=");
	printf("done");
	for (int i = 0; i < 17; i++)printf("=");
	printf("|\n"); fflush(stdout);
}

void checkErr(cudaError_t err) {
	if (cudaSuccess != err) {
		//printf("OpenCL error(%d)", err);
		printf("Err(%d)", err);
	}
	else {
		//printf("Success!");
		//printf(" - ");
	}
	//assert(err == CL_SUCCESS);
	fflush(stdout);
}

void monte_carlo::operator()(model& m, output_container& out, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	quasi_newton quasi_newton_par; const int quasi_newton_par_max_steps = ssd_par.evals;

	/**************************************************************************/
	/************************    Allocate CPU memory    ***********************/
	/**************************************************************************/
	printf("\nSearch depth is set to %d", search_depth);
	// Preparing m related data
	m_cl m_cl_cpu;
	assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

	for (int i = 0; i < m.atoms.size(); i++) {
		m_cl_cpu.atoms[i].types[0] = m.atoms[i].el;// To store 4 atoms types (el, ad, xs, sy)
		m_cl_cpu.atoms[i].types[1] = m.atoms[i].ad;
		m_cl_cpu.atoms[i].types[2] = m.atoms[i].xs;
		m_cl_cpu.atoms[i].types[3] = m.atoms[i].sy;
		for (int j = 0; j < 3; j++) {
			m_cl_cpu.atoms[i].coords[j] = m.atoms[i].coords[j];// To store atom coords
		}
	}

	// To store atoms coords
	for (int i = 0; i < m.coords.size(); i++) {
		for (int j = 0; j < 3; j++) {
			m_cl_cpu.m_coords.coords[i][j] = m.coords[i].data[j];
		}
	}

	//To store minus forces
	for (int i = 0; i < m.coords.size(); i++) {
		for (int j = 0; j < 3; j++) {
			m_cl_cpu.minus_forces.coords[i][j] = m.minus_forces[i].data[j];
		}
	}

	// Preparing ligand data
	assert(m.num_other_pairs() == 0); // m.other_paris is not supported!
	assert(m.ligands.size() == 1); // Only one ligand supported!
	m_cl_cpu.ligand.pairs.num_pairs = m.ligands[0].pairs.size();
	for (int i = 0; i < m_cl_cpu.ligand.pairs.num_pairs; i++) {
		m_cl_cpu.ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
		m_cl_cpu.ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
		m_cl_cpu.ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
	}
	m_cl_cpu.ligand.begin = m.ligands[0].begin; // 0
	m_cl_cpu.ligand.end = m.ligands[0].end; // 29
	ligand m_ligand = m.ligands[0]; // Only support one ligand 
	assert(m_ligand.end < MAX_NUM_OF_ATOMS);

	// Store root node
	m_cl_cpu.ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
	m_cl_cpu.ligand.rigid.atom_range[0][1] = m_ligand.node.end;
	for (int i = 0; i < 3; i++) m_cl_cpu.ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];
	for (int i = 0; i < 9; i++) m_cl_cpu.ligand.rigid.orientation_m[0][i] = m_ligand.node.get_orientation_m().data[i];
	m_cl_cpu.ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
	m_cl_cpu.ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
	m_cl_cpu.ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
	m_cl_cpu.ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
	for (int i = 0; i < 3; i++) { m_cl_cpu.ligand.rigid.axis[0][i] = 0; m_cl_cpu.ligand.rigid.relative_axis[0][i] = 0; m_cl_cpu.ligand.rigid.relative_origin[0][i] = 0; }

	// Store children nodes (in depth-first order)
	struct tmp_struct {
		int start_index = 0;
		int parent_index = 0;
		void store_node(tree<segment>& child_ptr, rigid_cl& rigid) {
			start_index++; // start with index 1, index 0 is root node
			rigid.parent[start_index] = parent_index;
			rigid.atom_range[start_index][0] = child_ptr.node.begin;
			rigid.atom_range[start_index][1] = child_ptr.node.end;
			for (int i = 0; i < 9; i++) rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
			rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
			rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
			rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
			rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
			for (int i = 0; i < 3; i++) {
				rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
				rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
				rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
				rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
			}
			if (child_ptr.children.size() == 0) return;
			else {
				assert(start_index < MAX_NUM_OF_RIGID);
				int parent_index_tmp = start_index;
				for (int i = 0; i < child_ptr.children.size(); i++) {
					this->parent_index = parent_index_tmp; // Update parent index
					this->store_node(child_ptr.children[i], rigid);
				}
			}
		}
	};
	tmp_struct ts;
	for (int i = 0; i < m_ligand.children.size(); i++) {
		ts.parent_index = 0; // Start a new branch, whose parent is 0
		ts.store_node(m_ligand.children[i], m_cl_cpu.ligand.rigid);
	}
	m_cl_cpu.ligand.rigid.num_children = ts.start_index;

	// set children_map
	for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
		for (int j = 0; j < MAX_NUM_OF_RIGID; j++)
			m_cl_cpu.ligand.rigid.children_map[i][j] = false;
	for (int i = 1; i < m_cl_cpu.ligand.rigid.num_children + 1; i++) {
		int parent_index = m_cl_cpu.ligand.rigid.parent[i];
		m_cl_cpu.ligand.rigid.children_map[parent_index][i] = true;
	}
	m_cl_cpu.m_num_movable_atoms = m.num_movable_atoms();
	size_t m_cl_cpu_size = sizeof(m_cl_cpu);

	// Preparing ig related data
	ig_cl* ig_cl_ptr = (ig_cl*)malloc(sizeof(ig_cl));
	ig_cl_ptr->atu = ig.get_atu(); // atu
	ig_cl_ptr->slope = ig.get_slope(); // slope
	std::vector<grid> tmp_grids = ig.get_grids();
	int grid_size = tmp_grids.size();
	assert(GRIDS_SIZE == grid_size); // grid_size has to be 17

	for (int i = 0; i < grid_size; i++) {
		for (int j = 0; j < 3; j++) {
			ig_cl_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
			ig_cl_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
			ig_cl_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grids[i].m_dim_fl_minus_1[j];
			ig_cl_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
		}
		if (tmp_grids[i].m_data.dim0() != 0) {
			ig_cl_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0(); assert(MAX_NUM_OF_GRID_MI >= ig_cl_ptr->grids[i].m_i);
			ig_cl_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1(); assert(MAX_NUM_OF_GRID_MJ >= ig_cl_ptr->grids[i].m_j);
			ig_cl_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2(); assert(MAX_NUM_OF_GRID_MK >= ig_cl_ptr->grids[i].m_k);

			for (int j = 0; j < ig_cl_ptr->grids[i].m_i * ig_cl_ptr->grids[i].m_j * ig_cl_ptr->grids[i].m_k; j++) {
				ig_cl_ptr->grids[i].m_data[j] = tmp_grids[i].m_data.m_data[j];
			}
			//int m_i = tmp_grids[i].m_data.dim0();
			//int m_j = tmp_grids[i].m_data.dim1();
			//int m_k = tmp_grids[i].m_data.dim2();
			//for(sz idx0 = 0; idx0 < m_i - 1; idx0++)
			//	for(sz idx1 = 0; idx1 < m_j - 1; idx1++)
			//		for (sz idx2 = 0; idx2 < m_k - 1; idx2++) {
			//			int base = (idx0 + m_i * (idx1 + m_j * idx2))*8;
			//			ig_cl_ptr->grids[i].m_data[base] = tmp_grids[i].m_data(idx0, idx1, idx2); // f000
			//			ig_cl_ptr->grids[i].m_data[base + 1] = tmp_grids[i].m_data(idx0+1, idx1, idx2); // f100
			//			ig_cl_ptr->grids[i].m_data[base + 2] = tmp_grids[i].m_data(idx0, idx1+1, idx2); // f010
			//			ig_cl_ptr->grids[i].m_data[base + 3] = tmp_grids[i].m_data(idx0+1, idx1+1, idx2); // f110
			//			ig_cl_ptr->grids[i].m_data[base + 4] = tmp_grids[i].m_data(idx0, idx1, idx2+1); // f001
			//			ig_cl_ptr->grids[i].m_data[base + 5] = tmp_grids[i].m_data(idx0+1, idx1, idx2+1); // f101
			//			ig_cl_ptr->grids[i].m_data[base + 6] = tmp_grids[i].m_data(idx0, idx1+1, idx2+1); // f011
			//			ig_cl_ptr->grids[i].m_data[base + 7] = tmp_grids[i].m_data(idx0+1, idx1+1, idx2+1); // f111
			//		}

		}
		else {
			ig_cl_ptr->grids[i].m_i = 0;
			ig_cl_ptr->grids[i].m_j = 0;
			ig_cl_ptr->grids[i].m_k = 0;
		}
	}
	size_t ig_cl_size = sizeof(*ig_cl_ptr);


	// Generating random ligand structures
	std::vector<output_type_cl*> rand_molec_struc_vec; rand_molec_struc_vec.resize(thread);
	for (int i = 0; i < thread; i++) {
		rand_molec_struc_vec[i] = (output_type_cl*)malloc(sizeof(output_type_cl));
	}

	//std::vector<output_type_cl> rand_molec_struc_vec; 
	//rand_molec_struc_vec.resize(thread);
	int lig_torsion_size = tmp.c.ligands[0].torsions.size();
	int flex_torsion_size; if (tmp.c.flex.size() != 0) flex_torsion_size = tmp.c.flex[0].torsions.size(); else flex_torsion_size = 0;
	std::vector<vec> uniform_data;
	uniform_data.resize(thread);

	for (int i = 0; i < thread; i++) {
		tmp.c.randomize(corner1, corner2, generator); // generate a random structure
		//Generate positions with uniform probability
		//generate_uniform_position(corner1, corner2, uniform_data, exhaustiveness);
		//for (int j = 0; j < 3; j++) rand_molec_struc_vec[i].position[j] = uniform_data[i].data[j];
		for (int j = 0; j < 3; j++) rand_molec_struc_vec[i]->position[j] = tmp.c.ligands[0].rigid.position[j];
		assert(lig_torsion_size < MAX_NUM_OF_LIG_TORSION);
		for (int j = 0; j < lig_torsion_size; j++) rand_molec_struc_vec[i]->lig_torsion[j] = tmp.c.ligands[0].torsions[j];// Only support one ligand
		assert(flex_torsion_size < MAX_NUM_OF_FLEX_TORSION);
		for (int j = 0; j < flex_torsion_size; j++) rand_molec_struc_vec[i]->flex_torsion[j] = tmp.c.flex[0].torsions[j];// Only support one flex

		rand_molec_struc_vec[i]->orientation[0] = (float)tmp.c.ligands[0].rigid.orientation.R_component_1();
		rand_molec_struc_vec[i]->orientation[1] = (float)tmp.c.ligands[0].rigid.orientation.R_component_2();
		rand_molec_struc_vec[i]->orientation[2] = (float)tmp.c.ligands[0].rigid.orientation.R_component_3();
		rand_molec_struc_vec[i]->orientation[3] = (float)tmp.c.ligands[0].rigid.orientation.R_component_4();

		rand_molec_struc_vec[i]->lig_torsion_size = lig_torsion_size;
	}

	// Preaparing p related data
	p_cl p_cl_cpu;
	p_cl_cpu.m_cutoff_sqr = p.cutoff_sqr();
	p_cl_cpu.factor = p.factor;
	p_cl_cpu.n = p.n;
	assert(MAX_P_DATA_M_DATA_SIZE > p.data.m_data.size());
	for (int i = 0; i < p.data.m_data.size(); i++) {
		p_cl_cpu.m_data[i].factor = p.data.m_data[i].factor;
		assert(FAST_SIZE == p.data.m_data[i].fast.size());
		assert(SMOOTH_SIZE == p.data.m_data[i].smooth.size());
		for (int j = 0; j < FAST_SIZE; j++) {
			p_cl_cpu.m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++) {
			p_cl_cpu.m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			p_cl_cpu.m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	size_t p_cl_cpu_size = sizeof(p_cl_cpu);

	// Generate random maps
	random_maps* rand_maps = (random_maps*)malloc(sizeof(random_maps));
	for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
		rand_maps->int_map[i] = random_int(0, int(lig_torsion_size), generator);
		rand_maps->pi_map[i] = random_fl(-pi, pi, generator);
	}
	for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
		vec rand_coords = random_inside_sphere(generator);
		for (int j = 0; j < 3; j++) {
			rand_maps->sphere_map[i][j] = rand_coords[j];
		}
	}
	size_t rand_maps_size = sizeof(*rand_maps);


	float hunt_cap_float[3] = { hunt_cap[0], hunt_cap[1], hunt_cap[2] };
	float authentic_v_float[3] = { authentic_v[0],authentic_v[1], authentic_v[2] };
	float mutation_amplitude_float = mutation_amplitude;
	float epsilon_fl_float = epsilon_fl;
	cudaError_t err;
	err = cudaSetDevice(0); checkErr(err);
 
/**************************************************************************/
/************************    Allocate GPU memory    ***********************/
/**************************************************************************/

	float* rand_molec_struc_vec_gpu;
	err = cudaMalloc((void**)&rand_molec_struc_vec_gpu, thread * SIZE_OF_MOLEC_STRUC); checkErr(err);
	for (int i = 0; i < thread; i++) {
		std::vector<float> pos(rand_molec_struc_vec[i]->position, rand_molec_struc_vec[i]->position + 3);
		std::vector<float> ori(rand_molec_struc_vec[i]->orientation, rand_molec_struc_vec[i]->orientation + 4);
		std::vector<float> lig_tor(rand_molec_struc_vec[i]->lig_torsion, rand_molec_struc_vec[i]->lig_torsion + MAX_NUM_OF_LIG_TORSION);
		std::vector<float> flex_tor(rand_molec_struc_vec[i]->flex_torsion, rand_molec_struc_vec[i]->flex_torsion + MAX_NUM_OF_FLEX_TORSION);
		float lig_tor_size = rand_molec_struc_vec[i]->lig_torsion_size;
		err = cudaMemcpy(rand_molec_struc_vec_gpu + i * (3+4+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION+1), pos.data(),
			pos.size() * sizeof(float), cudaMemcpyHostToDevice); checkErr(err);
		err = cudaMemcpy(rand_molec_struc_vec_gpu + i * (3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) + pos.size(),
			ori.data(), ori.size() * sizeof(float), cudaMemcpyHostToDevice); checkErr(err);
		err = cudaMemcpy(rand_molec_struc_vec_gpu + i * (3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) + pos.size() + ori.size(),
			lig_tor.data(), lig_tor.size() * sizeof(float), cudaMemcpyHostToDevice); checkErr(err);
		err = cudaMemcpy(rand_molec_struc_vec_gpu + i * (3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) + pos.size() + ori.size() + MAX_NUM_OF_LIG_TORSION,
			flex_tor.data(), flex_tor.size() * sizeof(float), cudaMemcpyHostToDevice); checkErr(err);
		err = cudaMemcpy(rand_molec_struc_vec_gpu + i * (3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) + pos.size() + ori.size() + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION,
			&lig_tor_size, sizeof(float), cudaMemcpyHostToDevice); checkErr(err);
	}

	random_maps* rand_maps_gpu;
	err = cudaMalloc(&rand_maps_gpu, rand_maps_size); checkErr(err);
	err = cudaMemcpy(rand_maps_gpu, rand_maps, rand_maps_size, cudaMemcpyHostToDevice); checkErr(err);

	// Preparing m related data
	m_cl* m_cl_gpu;
	err = cudaMalloc(&m_cl_gpu, thread * m_cl_cpu_size); checkErr(err);
	for(int i=0;i<thread;i++)
		err = cudaMemcpy(m_cl_gpu + i, &m_cl_cpu, m_cl_cpu_size, cudaMemcpyHostToDevice); checkErr(err);

	// Preparing p related data
	p_cl* p_cl_gpu;
	err = cudaMalloc(&p_cl_gpu, p_cl_cpu_size); checkErr(err);
	err = cudaMemcpy(p_cl_gpu, &p_cl_cpu, p_cl_cpu_size, cudaMemcpyHostToDevice); checkErr(err);

	// Preparing ig related data (cache related data)
	__constant__ ig_cl* ig_cl_gpu;
	err = cudaMalloc(&ig_cl_gpu, ig_cl_size); checkErr(err);
	err = cudaMemcpy(ig_cl_gpu, ig_cl_ptr, ig_cl_size, cudaMemcpyHostToDevice); checkErr(err);

	// Preparing result data
	output_type_cl* results;
	err = cudaMalloc(&results, thread * sizeof(output_type_cl)); checkErr(err);

/**************************************************************************/
/************************  Call to kernel function  ***********************/
/**************************************************************************/
	// Call to kernel function
	kernel2(rand_molec_struc_vec_gpu, rand_maps_gpu, m_cl_gpu, p_cl_gpu,ig_cl_gpu, results, search_depth, epsilon_fl_float, thread);

	output_type_cl* results_ptr = (output_type_cl*)malloc(thread * sizeof(output_type_cl));

	err = cudaMemcpy(results_ptr, results, thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost); checkErr(err);

	std::vector<output_type> result_vina = cl_to_vina(results_ptr, thread);

	err = cudaFree(rand_molec_struc_vec_gpu); checkErr(err);
	err = cudaFree(rand_maps_gpu); checkErr(err);
	err = cudaFree(m_cl_gpu); checkErr(err);
	err = cudaFree(p_cl_gpu); checkErr(err);
	err = cudaFree(ig_cl_gpu); checkErr(err);
	err = cudaFree(results); checkErr(err);

	// Write back to vina
	for (int i = 0; i < thread; i++) {
		add_to_output_container(out, result_vina[i], min_rmsd, num_saved_mins);
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e);

	free(ig_cl_ptr);
	free(rand_maps);
	for (int i = 0; i < thread; i++)free(rand_molec_struc_vec[i]);

}
#endif
