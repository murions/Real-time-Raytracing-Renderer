#ifndef CUDABUFFER_H
#define CUDABUFFER_H

#include <vector>
#include "optix_add.h"
#include <assert.h>

class CUDABuffer
{
public:
	void* d_ptr{ nullptr };
	size_t sizeInBytes{ 0 };
public:
	inline CUdeviceptr d_pointer() const {
		return (CUdeviceptr)d_ptr;
	}
	void resize(size_t size) {
		if (d_ptr) free();
		alloc(size);
	}
	void alloc(size_t size) {
		if (d_ptr) free();
		assert(d_ptr == nullptr);
		this->sizeInBytes = size;
		CUDA_CHECK(Malloc((void**)&d_ptr, sizeInBytes));
	}
	void free() {
		CUDA_CHECK(Free(d_ptr));
		d_ptr = nullptr;
		sizeInBytes = 0;
	}
	template<typename T> void alloc_and_upload(const std::vector<T>& vt) {
		alloc(vt.size() * sizeof(T));
		upload((const T*)vt.data(), vt.size());
	}
	template<typename T> void upload(const T* t, size_t count) {
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count * sizeof(T));
		CUDA_CHECK(Memcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
	}
	template<typename T> void download(T* t, size_t count) {
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count * sizeof(T));
		CUDA_CHECK(Memcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
	}
};

#endif // CUDABUFFER_H
