// Zero-copy dma-buf path for libxdna_sr's fsr1 frame-layout backend: two Vulkan host-visible,
// dma-buf-exportable buffers stand in for gamescope's swapchain images. No GPU shader runs -- the
// CPU fills the input through the Vulkan mapping and reads the output the same way; only the
// dma-buf export/import plumbing and the NPU dispatch are under test.
//
// Picks the AMD Radeon 890M (RADV) explicitly; the box also has an NVIDIA RTX eGPU that must not
// be used for buffer allocation (an NVIDIA dma-buf export would not import into amdxdna).
//
// Build:
//   g++ -std=c++17 -O2 dmabuf_frame_test.cpp -I../include -L<target/release> -lxdna_sr -lvulkan \
//       -Wl,-rpath,<target/release> -o dmabuf_frame_test
//
// usage: dmabuf_frame_test <fsr1.json> <in.rgb WxH> <ref.rgb> [4k-fsr1.json]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "xdna_sr.h"

namespace {

#define VK(x)                                                                          \
	do {                                                                            \
		VkResult r_ = (x);                                                      \
		if (r_ != VK_SUCCESS)                                                  \
			throw std::runtime_error(std::string(#x) + " = " + std::to_string(r_)); \
	} while (0)

double now_ms()
{
	timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

std::vector<uint8_t> slurp(const std::string &p)
{
	std::ifstream f(p, std::ios::binary | std::ios::ate);
	if (!f)
		throw std::runtime_error("cannot read " + p);
	std::vector<uint8_t> v(f.tellg());
	f.seekg(0);
	f.read(reinterpret_cast<char *>(v.data()), v.size());
	return v;
}

double median(std::vector<double> v)
{
	std::sort(v.begin(), v.end());
	return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Gpu {
	VkInstance inst{};
	VkPhysicalDevice pd{};
	VkDevice dev{};
	VkPhysicalDeviceMemoryProperties mp{};
	PFN_vkGetMemoryFdKHR getMemoryFd{};

	void init()
	{
		VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
		ai.apiVersion = VK_API_VERSION_1_3;
		VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ici.pApplicationInfo = &ai;
		VK(vkCreateInstance(&ici, nullptr, &inst));

		uint32_t n = 0;
		vkEnumeratePhysicalDevices(inst, &n, nullptr);
		std::vector<VkPhysicalDevice> pds(n);
		vkEnumeratePhysicalDevices(inst, &n, pds.data());
		for (auto p : pds) {
			VkPhysicalDeviceProperties pr;
			vkGetPhysicalDeviceProperties(p, &pr);
			if (std::string(pr.deviceName).find("890M") != std::string::npos) {
				pd = p;
				printf("gpu: %s\n", pr.deviceName);
			}
		}
		if (!pd)
			throw std::runtime_error("no 890M Vulkan device found (refusing the NVIDIA eGPU)");
		vkGetPhysicalDeviceMemoryProperties(pd, &mp);

		uint32_t nqf = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
		float prio = 1.0f;
		VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qci.queueFamilyIndex = 0;
		qci.queueCount = 1;
		qci.pQueuePriorities = &prio;
		const char *exts[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
				      VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
		VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		dci.enabledExtensionCount = 2;
		dci.ppEnabledExtensionNames = exts;
		VK(vkCreateDevice(pd, &dci, nullptr, &dev));
		getMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
	}

	uint32_t pick(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
	{
		for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
			auto f = mp.memoryTypes[i].propertyFlags;
			if ((bits & (1u << i)) && (f & want) == want && !(f & avoid))
				return i;
		}
		throw std::runtime_error("no suitable memory type");
	}
};

struct Buf {
	VkBuffer b{};
	VkDeviceMemory m{};
	int fd = -1;
	uint8_t *map = nullptr;
	size_t size = 0;
};

// Host-visible, dma-buf-exportable, NOT device-local/host-cached -- the "gtt" memory class that
// gpu_npu_roundtrip.cpp exercises for the GPU<->NPU path, so writes are visible with no explicit
// flush and the fd imports cleanly into amdxdna.
Buf make_buf(Gpu &g, size_t size)
{
	Buf r;
	r.size = size;
	VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
	ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.pNext = &ext;
	bci.size = size;
	bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VK(vkCreateBuffer(g.dev, &bci, nullptr, &r.b));

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g.dev, r.b, &req);

	VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
	exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
	ded.buffer = r.b;
	exp.pNext = &ded;
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.pNext = &exp;
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = g.pick(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
	VK(vkAllocateMemory(g.dev, &mai, nullptr, &r.m));
	VK(vkBindBufferMemory(g.dev, r.b, r.m, 0));

	VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
	gi.memory = r.m;
	gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VK(g.getMemoryFd(g.dev, &gi, &r.fd));

	VK(vkMapMemory(g.dev, r.m, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&r.map)));
	return r;
}

// Pad + edge-replicate an interleaved RGB8 frame into a BGRA frame, matching
// Fsr1FrameEngine::upscale_bgra8's host-side pad exactly (row/col replication at the frame edge).
void pad_bgra(const uint8_t *rgb, size_t w, size_t h, uint8_t *out, const XdnaSrFrameLayout &l)
{
	for (size_t r = 0; r < l.in_pad_h; r++) {
		size_t sy = r < l.in_pad_y ? 0 : std::min(r - l.in_pad_y, h - 1);
		uint8_t *row = out + r * l.in_pad_w * 4;
		for (size_t x = 0; x < l.in_pad_w; x++) {
			size_t sx = x < l.in_pad_x ? 0 : std::min(x - l.in_pad_x, w - 1);
			const uint8_t *p = rgb + (sy * w + sx) * 3;
			uint8_t *o = row + x * 4;
			o[0] = p[2];
			o[1] = p[1];
			o[2] = p[0];
			o[3] = 0xff;
		}
	}
}

void bgra_to_rgb(const uint8_t *bgra, size_t n_px, uint8_t *rgb)
{
	for (size_t i = 0; i < n_px; i++) {
		rgb[3 * i + 0] = bgra[4 * i + 2];
		rgb[3 * i + 1] = bgra[4 * i + 1];
		rgb[3 * i + 2] = bgra[4 * i + 0];
	}
}

void check(const char *what, int rc)
{
	if (rc != 0)
		throw std::runtime_error(std::string(what) + ": " + xdna_sr_last_error());
}

} // namespace

int main(int argc, char **argv)
{
	if (argc < 5) {
		fprintf(stderr, "usage: %s fsr1.json in.rgb WxH ref.rgb [4k-fsr1.json]\n", argv[0]);
		return 2;
	}
	std::string sched_path = argv[1], in_path = argv[2];
	size_t src_w, src_h;
	if (sscanf(argv[3], "%zux%zu", &src_w, &src_h) != 2)
		throw std::runtime_error("bad WxH");
	std::string ref_path = argv[4];
	const char *sched_4k = argc > 5 ? argv[5] : nullptr;

	Gpu g;
	g.init();

	XdnaSr *h = xdna_sr_create(sched_path.c_str(), 1);
	if (!h)
		throw std::runtime_error(std::string("xdna_sr_create: ") + xdna_sr_last_error());

	XdnaSrFrameLayout layout{};
	check("xdna_sr_frame_layout", xdna_sr_frame_layout(h, &layout));
	printf("layout: in %zux%zu, padded_in %zux%zu @(%zu,%zu), padded_out %zux%zu, scale %zu\n",
	       layout.in_w, layout.in_h, layout.in_pad_w, layout.in_pad_h, layout.in_pad_x,
	       layout.in_pad_y, layout.out_pad_w, layout.out_pad_h, layout.scale);
	if (layout.in_w != src_w || layout.in_h != src_h)
		throw std::runtime_error("schedule size does not match the test frame");

	auto src_rgb = slurp(in_path);
	if (src_rgb.size() != src_w * src_h * 3)
		throw std::runtime_error("input frame size mismatch");

	Buf in_buf = make_buf(g, layout.in_pad_w * layout.in_pad_h * 4);
	Buf out_buf = make_buf(g, layout.out_pad_w * layout.out_pad_h * 4);
	pad_bgra(src_rgb.data(), src_w, src_h, in_buf.map, layout);

	int fence = 0;
	check("xdna_sr_process_dmabuf", xdna_sr_process_dmabuf(h, in_buf.fd, out_buf.fd, &fence));
	if (fence != -1)
		throw std::runtime_error("out_fence_fd must be -1 in this version");

	size_t ow = layout.scale * src_w, oh = layout.scale * src_h;
	std::vector<uint8_t> got_rgb(ow * oh * 3);
	std::vector<uint8_t> row_rgb(ow * 3);
	for (size_t y = 0; y < oh; y++) {
		bgra_to_rgb(out_buf.map + y * layout.out_pad_w * 4, ow, row_rgb.data());
		memcpy(&got_rgb[y * ow * 3], row_rgb.data(), ow * 3);
	}

	auto ref_rgb = slurp(ref_path);
	if (ref_rgb.size() != got_rgb.size())
		throw std::runtime_error("reference size mismatch: got " + std::to_string(got_rgb.size()) +
					 ", ref " + std::to_string(ref_rgb.size()));
	bool identical = ref_rgb == got_rgb;
	printf("byte-identical vs reference: %s\n", identical ? "YES" : "NO");
	if (!identical) {
		size_t nbad = 0, first = SIZE_MAX;
		for (size_t i = 0; i < ref_rgb.size(); i++)
			if (ref_rgb[i] != got_rgb[i]) {
				nbad++;
				if (first == SIZE_MAX)
					first = i;
			}
		printf("  %zu/%zu bytes differ, first at %zu (got %d want %d)\n", nbad, ref_rgb.size(),
		       first, got_rgb[first], ref_rgb[first]);
	}

	// --- timing: dma-buf path vs the copy path, same design ---
	const int warm = 5, iters = 100;
	std::vector<double> t_dmabuf;
	for (int i = 0; i < warm + iters; i++) {
		double t0 = now_ms();
		check("process_dmabuf(timing)", xdna_sr_process_dmabuf(h, in_buf.fd, out_buf.fd, &fence));
		double t1 = now_ms();
		if (i >= warm)
			t_dmabuf.push_back(t1 - t0);
	}

	std::vector<uint8_t> copy_in(src_w * src_h * 4), copy_out(ow * oh * 4);
	for (size_t i = 0; i < src_w * src_h; i++) {
		copy_in[4 * i + 0] = src_rgb[3 * i + 2];
		copy_in[4 * i + 1] = src_rgb[3 * i + 1];
		copy_in[4 * i + 2] = src_rgb[3 * i + 0];
		copy_in[4 * i + 3] = 0xff;
	}
	std::vector<double> t_copy;
	size_t gow = 0, goh = 0;
	for (int i = 0; i < warm + iters; i++) {
		double t0 = now_ms();
		check("process_bgra8(timing)",
		      xdna_sr_process_bgra8(h, copy_in.data(), src_w, src_h, src_w * 4, copy_out.data(),
					    ow * 4, copy_out.size(), &gow, &goh));
		double t1 = now_ms();
		if (i >= warm)
			t_copy.push_back(t1 - t0);
	}
	printf("dma-buf path : median %.3f ms/frame over %d iters\n", median(t_dmabuf), iters);
	printf("copy path    : median %.3f ms/frame over %d iters\n", median(t_copy), iters);

	xdna_sr_free(h);

	if (sched_4k) {
		XdnaSr *h4 = xdna_sr_create(sched_4k, 1);
		if (!h4)
			throw std::runtime_error(std::string("xdna_sr_create(4k): ") + xdna_sr_last_error());
		XdnaSrFrameLayout l4{};
		check("xdna_sr_frame_layout(4k)", xdna_sr_frame_layout(h4, &l4));
		Buf in4 = make_buf(g, l4.in_pad_w * l4.in_pad_h * 4);
		Buf out4 = make_buf(g, l4.out_pad_w * l4.out_pad_h * 4);
		memset(in4.map, 0x40, in4.size);
		std::vector<double> t4;
		for (int i = 0; i < warm + iters; i++) {
			double t0 = now_ms();
			check("process_dmabuf(4k timing)", xdna_sr_process_dmabuf(h4, in4.fd, out4.fd, &fence));
			double t1 = now_ms();
			if (i >= warm)
				t4.push_back(t1 - t0);
		}
		printf("4k dma-buf (%zux%zu -> %zux%zu): median %.3f ms/frame over %d iters\n", l4.in_w,
		       l4.in_h, l4.scale * l4.in_w, l4.scale * l4.in_h, median(t4), iters);
		xdna_sr_free(h4);
	}

	return identical ? 0 : 1;
}
