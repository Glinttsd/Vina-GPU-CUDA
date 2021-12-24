extern "C"
void kernel2(float* rand_molec_struc_vec_gpu, random_maps * rand_maps_gpu, m_cl * m_cl_gpu, 
	p_cl * p_cl_gpu, ig_cl * ig_cl_gpu, output_type_cl* results, int search_depth,float epsilon_fl,int thread);