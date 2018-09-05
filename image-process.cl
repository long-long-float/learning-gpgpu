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


