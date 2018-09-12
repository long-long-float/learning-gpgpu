#define WORKGROUP_COL 32
#define WORKGROUP_ROW 32

kernel void grayscale(const global uchar* restrict image,
                   global uchar* result,
                   uint width, uint height) {
    /* printf("%d %d %d\n", get_global_id(0), get_group_id(0), get_local_id(0)); */
    uint x = get_global_id(0),
         y = get_global_id(1);

    uint index = (y * width + x) * 3;
    uchar gray = 0.2126 * image[index + 0] + 0.7152 * image[index + 1] + 0.0722 * image[index + 2];
    result[index + 0] = gray;
    result[index + 1] = gray;
    result[index + 2] = gray;
}

kernel void edge(const global uchar* restrict image,
                   global uchar* result,
                   uint width, uint height) {
    uint x = get_global_id(0),
         y = get_global_id(1);

    if (x < 1 || (width - 1) <= x || y < 1 || (height - 1) <= y) return;

    uint index = (y * width + x) * 3;

    for (int i = 0; i < 3; i++) {
        int c =
            -image[index - 3 + i] + image[index + 3 + i] // x
            -image[index - width * 3 + i] + image[index + width * 3 + i]; // y
        result[index + i] = 255 - max(c, 0);
    }
}

kernel void edge_use_local_mem(const global uchar* restrict image,
                   global uchar* result,
                   uint width, uint height) {
    const uint x = get_global_id(0),
               y = get_global_id(1);
    const uint lx = get_local_id(0),
               ly = get_local_id(1);
    const uint lwidth = get_local_size(0),
               lheight = get_local_size(1);

    /* __local uchar local_image[lwidth * lheight]; */
    __local uchar local_image[(WORKGROUP_COL + 2) * (WORKGROUP_ROW + 2) * 3];

    if (x < 1 || (width - 1) <= x || y < 1 || (height - 1) <= y) return;

    const uint index = (y * width + x) * 3,
               lindex = (ly * lwidth + lx) * 3;

    for (int i = 0; i < 3; i++) {
        local_image[lindex + i] = image[index + i];
    }
    /* if (ly <= 1) {
        for (int i = 0; i < 3; i++) {
            local_image[((ly * lwidth * lheight) + lx) * 3 + i] = image[ + i];
        }
    } */
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++) {
        // ワープダイバージェンスが発生しないようにする
        int c =
            // x
            - (lx >= 1 ? local_image[lindex - 3 + i] : image[index - 3 + i])
            + (lx < lwidth - 1 ? local_image[lindex + 3 + i] : image[index + 3 + i])
            // y
            - (ly >= 1 ? local_image[lindex - lwidth * 3 + i] : image[index - width * 3 + i])
            + (ly < lheight - 1 ? local_image[lindex + lheight * 3 + i] : image[index + width * 3 + i]);
        result[index + i] = 255 - max(c, 0);
    }
}


