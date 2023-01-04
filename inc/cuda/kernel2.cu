#include "macro.cuh"
#include "mutate.cuh"
#include "bfgs.cuh"
#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
// #include "kernel2.cuh"

__global__ void cuda_print() {
	printf("hello cuda !");
}

__device__ void print_output(output_type_cl* a) {
	for (int i = 0; i < 3 + 4 + MAX_NUM_OF_LIG_TORSION;i++) {
		if (i < 3)printf("position[%d] = %f\n", i, a->position[i]);
		else if (i < 7)printf("orientation[%d] = %f\n", i - 3, a->orientation[i - 3]);
		else if (i < 3 + 4 + MAX_NUM_OF_LIG_TORSION)printf("lig_torsion[%d] = %f\n", i - 7, a->lig_torsion[i - 7]);
	}
	printf("\n");
}

__device__ void output_type_cl_init(output_type_cl* out, float* ptr,const int threadNumInBlock,const int threadsPerBlock) {
	//for (int i = 0; i < 3; i++)out->position[i] = ptr[i];
	//for (int i = 0; i < 4; i++)out->orientation[i] = ptr[i + 3];
	//for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out->lig_torsion[i] = ptr[i + 3 + 4];
	//for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out->flex_torsion[i] = ptr[i + 3 + 4 + MAX_NUM_OF_LIG_TORSION];
	//out->lig_torsion_size = ptr[3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION]; 
	//did not assign coords and e
	for (int i = threadNumInBlock;
		i < 3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1;
		i = i + threadsPerBlock
		)
	{
		if(0<i<3)
			out->position[i] = ptr[i];
		if(3<=i<7)
			out->orientation[i-3] = ptr[i];
		if (7 <= i < 7 + MAX_NUM_OF_LIG_TORSION)
			out->lig_torsion[i - 7] = ptr[i];
		if (7 + MAX_NUM_OF_LIG_TORSION <= i < 7 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION)
			out->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION] = ptr[i];
		if(i == 7 + MAX_NUM_OF_LIG_TORSION+ MAX_NUM_OF_FLEX_TORSION)
			out->lig_torsion_size = ptr[i];
	}
	//did not assign coords and e
}

/**************************************************************************/
/**************************    Kernel2 Related    *************************/
/**************************************************************************/

__device__ void get_heavy_atom_movable_coords(output_type_cl* tmp, const m_cl* m_cl_gpu, m_coords_cl* m_coords) {
	int counter = 0;
	for (int i = 0; i < m_cl_gpu->m_num_movable_atoms; i++) {
		if (m_cl_gpu->atoms[i].types[0] != EL_TYPE_H_CL) {
			for (int j = 0; j < 3; j++)tmp->coords[counter][j] = m_coords->coords[i][j];
			counter++;
		}
		else {
			//printf("\n kernel2: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	//assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++)tmp->coords[i][j] = 0;
	}
}

//Generate a random number according to step
__device__ float generate_n(float* pi_map, const int step) {
	return fabs(pi_map[step]) / M_PI;
}

__device__ bool metropolis_accept(float old_f, float new_f, float temperature, float n) {
	if (new_f < old_f)return true;
	const float acceptance_probability = exp((old_f - new_f) / temperature);
	bool res = n < acceptance_probability;
	return n < acceptance_probability;
}

__device__ void write_back(output_type_cl* results, const output_type_cl* best_out) {
	for (int i = 0; i < 3; i++)results->position[i] = best_out->position[i];
	for (int i = 0; i < 4; i++)results->orientation[i] = best_out->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)results->lig_torsion[i] = best_out->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)results->flex_torsion[i] = best_out->flex_torsion[i];
	results->lig_torsion_size = best_out->lig_torsion_size;
	results->e = best_out->e;
	for (int i = 0; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++) {
			results->coords[i][j] = best_out->coords[i][j];
		}
	}
}

__global__ void monte_carlo (	float *				rand_molec_struc_vec_gpu,
								random_maps*		rand_maps_gpu,
								m_cl*				m_cl_global,
								p_cl*				p_cl_gpu,
								ig_cl*				ig_cl_gpu,
								output_type_cl*		results,
								int					search_depth,
								float				epsilon_fl,
								int					thread
){  
	//int id = threadIdx.x + threadIdx.y * blockDim.x + (blockIdx.x + blockIdx.y * blockDim.x) * blockDim.y * blockDim.x;
	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;
	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	int id = blockNumInGrid;
	//printf("\nid = %d", threadNumInBlock);
	float mutation_amplitude = 2.0;
	int bfgs_max_steps = 19;
	float hunt_cap[3] = { 10,10,10 };
	float authentic_v_gpu[3] = { 1000,1000,1000 };
	float best_e = INFINITY;

	m_cl* m_cl_gpu = &m_cl_global[id];

	//if (id == 0) {
	//	m_cl_gpu->m_coords.coords[0][0] = 111;
	//	//m_cl_global[blockNumInGrid].m_coords.coords[0][0] = 111;
	//	printf("\n coords = %f", m_cl_global[blockNumInGrid].m_coords.coords[0][0]);
	//}

	//__syncthreads();

	__shared__ m_coords_cl m_coords;
	m_coords = m_cl_gpu->m_coords;
	__shared__ m_minus_forces minus_forces;
	minus_forces = m_cl_gpu->minus_forces;


	__shared__ output_type_cl tmp;
	float* ptr = rand_molec_struc_vec_gpu + id * (SIZE_OF_MOLEC_STRUC / sizeof(float));
	output_type_cl_init(&tmp, rand_molec_struc_vec_gpu + id * (SIZE_OF_MOLEC_STRUC / sizeof(float)), threadNumInBlock, threadsPerBlock);
	//printf("\nthreadsPerBlock = %d", threadsPerBlock);
	__syncthreads();
	//if (threadNumInBlock == 0)printf("\nlig_torsion_size = %f", tmp.lig_torsion_size);
	__shared__ change_cl g;
	g.lig_torsion_size = tmp.lig_torsion_size;

	__shared__ output_type_cl best_out;
	__shared__ output_type_cl candidate;
	//printf("epsilon_fl = %f", epsilon_fl);
	//printf("\nid=%d, coords=[%d]", i, ig_cl_gpu->grids[i].m_range[0]);
	//printf("\nid=%d, coords=[%f]", i, p_cl_gpu->m_data[0].smooth[i][0]);
	//print_output(&tmp);
	//printf("\nid = %d, pos[0] = %f, pos[1] = %f, pos[2] = %f", i, tmp.position[0], tmp.position[1], tmp.position[2]);
	//printf("\nid = %d, random = %d", i, rand_maps_gpu->int_map[i]);
	

	for (int step = 0; step < search_depth; step++) {
		// printf("\nstep = %d", step);
		candidate = tmp;
		int map_index = (step + id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
		//if(threadNumInBlock==0) print_output(&candidate);
		mutate_conf_cl(	map_index,
						&candidate,
						rand_maps_gpu->int_map,
						rand_maps_gpu->sphere_map,
						rand_maps_gpu->pi_map,
						//m_cl_gpu->ligand.begin,
						//m_cl_gpu->ligand.end,
						//m_cl_gpu->atoms,
						//&m_coords,
						//m_cl_gpu->ligand.rigid.origin[0],
						epsilon_fl,
						mutation_amplitude,
						threadNumInBlock,
						threadsPerBlock
			);
		//if (threadNumInBlock == 0) print_output(&candidate);
		__syncthreads();
		//if (threadNumInBlock == 0) {
		//	printf("\nHere!");
		//	printf("\n thread %d, tmp = %f", threadNumInBlock, tmp.position[0]);
		//}
		//int a = 0;
		bfgs(	&candidate,// shared memory
				&g,// shared memory
				m_cl_gpu, // global memory
				p_cl_gpu,
				ig_cl_gpu,
				hunt_cap,
				epsilon_fl,
				bfgs_max_steps,
				&m_coords,   // shared memory
				&minus_forces, // shared memory
				threadNumInBlock,
				threadsPerBlock
		);

		float n = generate_n(rand_maps_gpu->pi_map, map_index);

		if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n)) {

			tmp = candidate;

			set(&tmp, &m_cl_gpu->ligand.rigid, &m_coords,
				m_cl_gpu->atoms, m_cl_gpu->m_num_movable_atoms, epsilon_fl);

			if (tmp.e < best_e) {
				bfgs(&tmp,
					&g,
					m_cl_gpu,
					p_cl_gpu,
					ig_cl_gpu,
					authentic_v_gpu,
					epsilon_fl,
					bfgs_max_steps,
					&m_coords,   // shared memory
					&minus_forces, // shared memory
					threadNumInBlock,
					threadsPerBlock
				);
				// set
				if (tmp.e < best_e) {
					set(&tmp, &m_cl_gpu->ligand.rigid, &m_coords,
						m_cl_gpu->atoms, m_cl_gpu->m_num_movable_atoms, epsilon_fl);

					best_out = tmp;
					get_heavy_atom_movable_coords(&best_out, m_cl_gpu, &m_coords); // get coords
					best_e = tmp.e;
				}
			}
		}
		
	}
	results[id] = best_out;
}





extern "C"
void kernel2(	float *				rand_molec_struc_vec_gpu, 
				random_maps *		rand_maps_gpu,
				m_cl *				m_cl_gpu,
				p_cl *				p_cl_gpu,
				ig_cl *				ig_cl_gpu,
				output_type_cl *	results,
				int					search_depth,
				int					num_steps,
				float				epsilon_fl,
				int					thread
) {
	// size_t global_size[2] = { 512, 32 };
	// size_t local_size[2] = { 16,8 };
	dim3 block_size(8, 8);
	dim3 grid_size(GRID_DIM1, GRID_DIM2);
	int tmp = thread / 32;

	cudaEvent_t time1, time2;
	cudaEventCreate(&time1);	cudaEventCreate(&time2);//�����¼�

	cudaEventRecord(time1, 0);//��¼ʱ���

	monte_carlo << <grid_size, block_size >>>(rand_molec_struc_vec_gpu, rand_maps_gpu, m_cl_gpu,
		p_cl_gpu, ig_cl_gpu, results, search_depth, epsilon_fl, thread);



	cudaError_t error;
	float kernalExecutionTime;
	error = cudaGetLastError(); 
	if (error != cudaSuccess) 
	{
		printf("%s\n", cudaGetErrorString(error)); system("pause");
	}
	cudaEventRecord(time2, 0);

	cudaEventSynchronize(time1);	cudaEventSynchronize(time2);//�ȴ�ʱ���ǰ���߳�ͬ��

	cudaEventElapsedTime(&kernalExecutionTime, time1, time2);//����ʱ���
	printf("\nElapsed time for GPU calculation: %0.3f s \n", kernalExecutionTime/1000);//���

	cudaEventDestroy(time1);	cudaEventDestroy(time2);//�����¼�


	//getchar();

}