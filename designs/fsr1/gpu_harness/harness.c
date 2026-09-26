/* Headless Vulkan compute harness: runs shaders/easu.comp then shaders/rcas.comp
 * on a raw RGBA8 input, dumps the RGBA8 result. GPU ground truth for
 * designs/fsr1/cpu_ref.py -- this is the actual AMD FSR1 reference source
 * (ffx_a.h/ffx_fsr1.h), not a re-implementation.
 *
 * Usage: harness <in_w> <in_h> <in.rgba8> <out.rgba8> [easu_out.rgba8]
 * Fixed x3 integer scale: out_w=in_w*3, out_h=in_h*3.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len);
    if (fread(buf, 1, len, f) != (size_t)len) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    *out_len = len;
    return buf;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    fprintf(stderr, "no matching memory type\n");
    exit(1);
}

typedef struct { VkImage img; VkDeviceMemory mem; VkImageView view; } GpuImage;

static GpuImage make_image(VkDevice dev, VkPhysicalDevice phys, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    GpuImage gi = {0};
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = (VkExtent3D){w, h, 1};
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHK(vkCreateImage(dev, &ici, NULL, &gi.img));
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, gi.img, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHK(vkAllocateMemory(dev, &mai, NULL, &gi.mem));
    CHK(vkBindImageMemory(dev, gi.img, gi.mem, 0));
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = gi.img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CHK(vkCreateImageView(dev, &vci, NULL, &gi.view));
    return gi;
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *mapped; } HostBuf;

static HostBuf make_host_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage) {
    HostBuf hb = {0};
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHK(vkCreateBuffer(dev, &bci, NULL, &hb.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, hb.buf, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(phys, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHK(vkAllocateMemory(dev, &mai, NULL, &hb.mem));
    CHK(vkBindBufferMemory(dev, hb.buf, hb.mem, 0));
    CHK(vkMapMemory(dev, hb.mem, 0, size, 0, &hb.mapped));
    return hb;
}

/* AU1_AF1 bit-cast, matches ffx_a.h. */
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s in_w in_h in.rgba8 out.rgba8 [easu_out.rgba8]\n", argv[0]); return 1; }
    uint32_t in_w = atoi(argv[1]), in_h = atoi(argv[2]);
    uint32_t out_w = in_w * 3, out_h = in_h * 3;
    const char *in_path = argv[3], *out_path = argv[4];
    const char *easu_dump_path = argc > 5 ? argv[5] : NULL;

    size_t in_len;
    char *in_pixels = read_file(in_path, &in_len);
    if (in_len != (size_t)in_w * in_h * 4) { fprintf(stderr, "input size mismatch: got %zu want %u\n", in_len, in_w*in_h*4); return 1; }

    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst; CHK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, NULL);
    if (!ndev) { fprintf(stderr, "no vulkan device\n"); return 1; }
    VkPhysicalDevice *devs = malloc(ndev * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &ndev, devs);
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    fprintf(stderr, "GPU: %s\n", props.deviceName);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qprops = malloc(nq * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qprops);
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    if (qfam == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo dqci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = qfam; dqci.queueCount = 1; dqci.pQueuePriorities = &qprio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqci;
    VkDevice dev; CHK(vkCreateDevice(phys, &dci, NULL, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfam, 0, &queue);

    GpuImage src = make_image(dev, phys, in_w, in_h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    GpuImage mid = make_image(dev, phys, out_w, out_h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    GpuImage dst = make_image(dev, phys, out_w, out_h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    VkDeviceSize in_size = (VkDeviceSize)in_w * in_h * 4;
    VkDeviceSize mid_size = (VkDeviceSize)out_w * out_h * 4;
    HostBuf upload = make_host_buf(dev, phys, in_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    memcpy(upload.mapped, in_pixels, in_size);
    HostBuf readback = make_host_buf(dev, phys, mid_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    HostBuf readback_mid = easu_dump_path ? make_host_buf(dev, phys, mid_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT) : (HostBuf){0};

    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.unnormalizedCoordinates = VK_FALSE;
    VkSampler sampler; CHK(vkCreateSampler(dev, &sci, NULL, &sampler));
    VkSamplerCreateInfo sci_nearest = sci; sci_nearest.magFilter = VK_FILTER_NEAREST; sci_nearest.minFilter = VK_FILTER_NEAREST;
    VkSampler sampler_nearest; CHK(vkCreateSampler(dev, &sci_nearest, NULL, &sampler_nearest));

    VkDescriptorSetLayoutBinding binds[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl; CHK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout; CHK(vkCreatePipelineLayout(dev, &plci, NULL, &playout));

    size_t easu_len, rcas_len;
    char *easu_spv = read_file("shaders/easu.spv", &easu_len);
    char *rcas_spv = read_file("shaders/rcas.spv", &rcas_len);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = easu_len; smci.pCode = (uint32_t*)easu_spv;
    VkShaderModule easu_mod; CHK(vkCreateShaderModule(dev, &smci, NULL, &easu_mod));
    smci.codeSize = rcas_len; smci.pCode = (uint32_t*)rcas_spv;
    VkShaderModule rcas_mod; CHK(vkCreateShaderModule(dev, &smci, NULL, &rcas_mod));

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = easu_mod; cpci.stage.pName = "main";
    cpci.layout = playout;
    VkPipeline easu_pipe; CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &easu_pipe));
    cpci.stage.module = rcas_mod;
    VkPipeline rcas_pipe; CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &rcas_pipe));

    VkDescriptorPoolSize dps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
    };
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2; dpci.poolSizeCount = 2; dpci.pPoolSizes = dps;
    VkDescriptorPool dpool; CHK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));

    VkDescriptorSetLayout layouts2[2] = {dsl, dsl};
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 2; dsai.pSetLayouts = layouts2;
    VkDescriptorSet dsets[2];
    CHK(vkAllocateDescriptorSets(dev, &dsai, dsets));

    VkCommandPoolCreateInfo cpci2 = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qfam;
    VkCommandPool cpool; CHK(vkCreateCommandPool(dev, &cpci2, NULL, &cpool));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHK(vkBeginCommandBuffer(cmd, &cbbi));

    /* src: UNDEFINED -> TRANSFER_DST -> copy from upload buf -> SHADER_READ */
    VkImageMemoryBarrier barr = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barr.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barr.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr.image = src.img; barr.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    barr.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,NULL,0,NULL,1,&barr);

    VkBufferImageCopy bic = {0};
    bic.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    bic.imageExtent = (VkExtent3D){in_w, in_h, 1};
    vkCmdCopyBufferToImage(cmd, upload.buf, src.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);

    barr.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barr.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barr.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barr.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,0,NULL,0,NULL,1,&barr);

    /* mid: UNDEFINED -> GENERAL (storage write target) */
    VkImageMemoryBarrier barr_mid = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barr_mid.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barr_mid.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barr_mid.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barr_mid.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr_mid.image = mid.img; barr_mid.subresourceRange = barr.subresourceRange;
    barr_mid.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,0,NULL,0,NULL,1,&barr_mid);
    VkImageMemoryBarrier barr_dst = barr_mid; barr_dst.image = dst.img;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,0,NULL,0,NULL,1,&barr_dst);

    /* descriptor writes: set0 = (src linear, mid storage) for EASU; set1 = (mid nearest, dst storage) for RCAS */
    VkDescriptorImageInfo src_info = {sampler, src.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo mid_storage_info = {VK_NULL_HANDLE, mid.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mid_sample_info = {sampler_nearest, mid.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dst_storage_info = {VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[4];
    w[0] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,dsets[0],0,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&src_info,NULL,NULL};
    w[1] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,dsets[0],1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&mid_storage_info,NULL,NULL};
    w[2] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,dsets[1],0,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&mid_sample_info,NULL,NULL};
    w[3] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,dsets[1],1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&dst_storage_info,NULL,NULL};
    vkUpdateDescriptorSets(dev, 4, w, 0, NULL);

    /* FsrEasuCon, computed on the CPU exactly like EasuPushData_t does. */
    float ow = (float)out_w, oh = (float)out_h, iw = (float)in_w, ih = (float)in_h;
    uint32_t con0[4] = { f2u(iw/ow), f2u(ih/oh), f2u(0.5f*iw/ow-0.5f), f2u(0.5f*ih/oh-0.5f) };
    uint32_t con1[4] = { f2u(1.0f/iw), f2u(1.0f/ih), f2u(1.0f/iw), f2u(-1.0f/ih) };
    uint32_t con2[4] = { f2u(-1.0f/iw), f2u(2.0f/ih), f2u(1.0f/iw), f2u(2.0f/ih) };
    uint32_t con3[4] = { f2u(0.0f), f2u(4.0f/ih), 0, 0 };
    uint32_t con_push[16];
    memcpy(con_push, con0, 16); memcpy(con_push+4, con1, 16);
    memcpy(con_push+8, con2, 16); memcpy(con_push+12, con3, 16);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, easu_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dsets[0], 0, NULL);
    vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, con_push);
    vkCmdDispatch(cmd, (out_w+7)/8, (out_h+7)/8, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,1,&mb,0,NULL,0,NULL);

    /* FsrRcasCon(sharpness=0.2), matches gamescope's g_upscaleFilterSharpness/10 default (2/10). */
    float sharpness = 0.2f;
    float rcas_scale = exp2f(-sharpness);
    uint32_t con_rcas[16] = {0};
    con_rcas[0] = f2u(rcas_scale);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rcas_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dsets[1], 0, NULL);
    vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, con_rcas);
    vkCmdDispatch(cmd, (out_w+7)/8, (out_h+7)/8, 1);

    VkImageMemoryBarrier barr_dst2 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barr_dst2.oldLayout = VK_IMAGE_LAYOUT_GENERAL; barr_dst2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barr_dst2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barr_dst2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr_dst2.image = dst.img; barr_dst2.subresourceRange = barr.subresourceRange;
    barr_dst2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barr_dst2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,NULL,0,NULL,1,&barr_dst2);

    VkBufferImageCopy bic_out = {0};
    bic_out.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    bic_out.imageExtent = (VkExtent3D){out_w, out_h, 1};
    vkCmdCopyImageToBuffer(cmd, dst.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &bic_out);

    if (easu_dump_path) {
        VkImageMemoryBarrier barr_mid2 = barr_dst2; barr_mid2.image = mid.img;
        barr_mid2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,NULL,0,NULL,1,&barr_mid2);
        vkCmdCopyImageToBuffer(cmd, mid.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_mid.buf, 1, &bic_out);
    }

    CHK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHK(vkQueueWaitIdle(queue));

    FILE *fo = fopen(out_path, "wb");
    fwrite(readback.mapped, 1, mid_size, fo);
    fclose(fo);
    if (easu_dump_path) {
        FILE *fm = fopen(easu_dump_path, "wb");
        fwrite(readback_mid.mapped, 1, mid_size, fm);
        fclose(fm);
    }
    fprintf(stderr, "wrote %s (%ux%u rgba8)\n", out_path, out_w, out_h);
    return 0;
}
