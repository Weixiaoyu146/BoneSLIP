def find_arithmetic_sequences(heights, N):
    """寻找可以形成列数为 N 的等差数列"""
    n = len(heights)
    sequences = []

    # 遍历每个可能的起点
    for i in range(n):
        for j in range(i + 1, n):
            diff = heights[j] - heights[i]
            current_sequence = [heights[i], heights[j]]

            # 从第 j + 1 个元素开始，寻找符合等差数列的元素
            for k in range(j + 1, n):
                if heights[k] - current_sequence[-1] == diff:
                    current_sequence.append(heights[k])

                # 如果找到了长度为 N 的等差数列，则记录
                if len(current_sequence) == N:
                    sequences.append(current_sequence[:])
                    break

            # 如果已找到足够的行，提前退出
            if len(sequences) * N >= n:
                break

        # 如果已找到足够的行，提前退出
        if len(sequences) * N >= n:
            break

    return sequences


def find_largest_matrix(heights):
    heights.sort()  # 将身高排序
    n = len(heights)

    # 从最大的可能列数 N 开始，逐渐减少
    for N in range(n, 1, -1):
        sequences = find_arithmetic_sequences(heights, N)
        if len(sequences) * N == n:
            # 输出结果
            print(f"lenth:  {N}")
            for row in sequences:
                print(row)
            return

    # 如果没有找到满足条件的方阵，则输出最小列数 1
    print("lenth:  1")
    print([height for height in heights])


# 输入部分
heights_input = input().strip()
heights = list(map(int, heights_input.split(", ")))

# 输出结果
find_largest_matrix(heights)