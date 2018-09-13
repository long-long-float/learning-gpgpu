#define LOCAL_WORK_COL 32
#define LOCAL_WORK_ROW 32

inline uint pos2idx(uint x, uint y, uint w) { return (y * w + x) * 3; }

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
    __local uchar local_image[(LOCAL_WORK_COL + 2) * (LOCAL_WORK_ROW + 2) * 3];

    if (x < 1 || (width - 1) <= x || y < 1 || (height - 1) <= y) return;

    for (int i = 0; i < 3; i++) {
        local_image[pos2idx(lx + 1, ly + 1, LOCAL_WORK_COL + 2) + i] = image[pos2idx(x, y, width) + i];
    }
    if (ly == 0) {
        for (int i = 0; i < 3; i++) local_image[pos2idx(1 + lx, 0, LOCAL_WORK_COL + 2) + i] = image[pos2idx(x, y - 1, width) + i];
    }
    if (ly == 1) {
        for (int i = 0; i < 3; i++) {
            local_image[pos2idx(1 + lx, LOCAL_WORK_ROW + 1, LOCAL_WORK_COL + 2) + i] = image[pos2idx(x, y + LOCAL_WORK_ROW - 1, width) + i];
        }
    }
    if (lx == 0) {
        for (int i = 0; i < 3; i++) local_image[pos2idx(0, 1 + ly, LOCAL_WORK_COL + 2) + i] = image[pos2idx(x - 1, y, width) + i];
    }
    if (lx == 1) {
        for (int i = 0; i < 3; i++) {
            local_image[pos2idx(LOCAL_WORK_COL + 1, 1 + ly, LOCAL_WORK_COL + 2) + i] = image[pos2idx(x + LOCAL_WORK_COL - 1, y, width) + i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++) {
        int c =
            // x
            - local_image[pos2idx(lx, ly + 1, LOCAL_WORK_COL + 2) + i]
            + local_image[pos2idx(lx + 2, ly + 1, LOCAL_WORK_COL + 2) + i]
            // y
            - local_image[pos2idx(lx + 1, ly, LOCAL_WORK_COL + 2) + i]
            + local_image[pos2idx(lx + 1, ly + 2, LOCAL_WORK_COL + 2) + i];
        result[pos2idx(x, y, width) + i] = 255 - max(c, 0);
    }
}


