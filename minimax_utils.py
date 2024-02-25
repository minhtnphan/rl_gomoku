import math


def get_horizontals(board, player):
    horizontals = ['' for i in range(len(board[0]))]
    for i in range(len(board[0])):
        for j in range(len(board[0])):
            if board[i][j] == 3 - player:
                horizontals[i] += 'x'
            elif board[i][j] == 0:
                horizontals[i] += 'o'
            elif board[i][j] == player:
                horizontals[i] += '*'
    return horizontals


def get_verticals(board, player):
    verticals = ['' for i in range(len(board))]
    for i in range(len(board[0])):
        for j in range(len(board)):
            if board[i][j] == 3 - player:
                verticals[j] += 'x'
            elif board[i][j] == 0:
                verticals[j] += 'o'
            elif board[i][j] == player:
                verticals[j] += '*'
    return verticals


def get_left_diags(board, player):
    size = len(board) if len(board) == len(board[0]) else 0
    centered_diag = math.floor((2 * size - 1 - 8) / 2)

    if size == 0:
        return []

    diags = ['']

    # main diag
    for i in range(size):
        if board[i][i] == 3 - player:
            diags[0] += 'x'
        elif board[i][i] == 0:
            diags[0] += 'o'
        elif board[i][i] == player:
            diags[0] += '*'

    # get diags from center to left (\)
    for i in range(centered_diag):
        d = ''
        for j in range(size):
            if (i + 1 + j) > size - 1 or board[i + 1 + j][j] == 3 - player:
                d += 'x'
            elif board[i + 1 + j][j] == 0:
                d += 'o'
            elif board[i + 1 + j][j] == player:
                d += '*'
        diags.append(d)

    # get diags from center to right (/)
    for i in range(centered_diag):
        d = ''
        for j in range(size):
            if (i + 1 + j) > size - 1 or board[j][i + 1 + j] == 3 - player:
                d += 'x'
            elif board[j][i + 1 + j] == 0:
                d += 'o'
            elif board[j][i + 1 + j] == player:
                d += '*'
        diags.append(d)

    return diags


def get_right_diags(board, player):
    size = len(board) if len(board) == len(board[0]) else 0
    centered_diag = math.floor((2 * size - 1 - 8) / 2)

    diags = ['']
    for i in range(size):
        if board[i][size - 1 - i] == 3 - player:
            diags[0] += 'x'
        elif board[i][size - 1 - i] == 0:
            diags[0] += 'o'
        elif board[i][size - 1 - i] == player:
            diags[0] += '*'

    for i in range(centered_diag):
        d = ''
        for j in range(size):
            if (centered_diag - i + j) > size - 1 or board[centered_diag - i + j][size - 1 - j] == 3 - player:
                d += 'x'
            elif board[centered_diag - i + j][size - 1 - j] == 0:
                d += 'o'
            elif board[centered_diag - i + j][size - 1 - j] == player:
                d += '*'
        diags.append(d)

    end_count = size - 2
    for i in range(centered_diag):
        d = ''
        for j in range(size):
            if (end_count - j) - i < 0 or board[j][(end_count - j) - i] == 3 - player:
                d += 'x'
            elif board[j][(end_count - j) - i] == 0:
                d += 'o'
            elif board[j][(end_count - j) - i] == player:
                d += '*'
        diags.append(d)

    return diags


def evaluate(board, player):
    new_board = board.copy()
    hori = get_horizontals(new_board, player)
    vert = get_verticals(new_board, player)
    left_D = get_left_diags(new_board, player)
    right_D = get_right_diags(new_board, player)
    allLines = hori + vert + left_D + right_D
    allLines = [line for line in allLines if line.count('*') > 1]

    score = 0

    for line in allLines:
        score += eval_line(line)
    return score


def eval_line(line):
    # * for piece placed, x for blocked square, o for open square

    five = '*****'

    open_four = 'o****o'

    closed_four1 = 'x****o'
    closed_four2 = 'o****x'
    closed_four3 = '*o***'
    closed_four4 = '***o*'
    closed_four5 = '**o**'

    open_three1 = 'o***oo'
    open_three2 = 'oo***o'
    open_three3 = 'o*o**o'
    open_three4 = 'o**o*o'

    closed_three1 = 'x***oo'
    closed_three2 = 'oo***x'
    closed_three3 = 'xo***ox'
    closed_three4 = 'o*o**x'
    closed_three5 = 'x*o**o'
    closed_three6 = 'x**o*o'
    closed_three6 = 'o**o*x'

    open_two1 = 'o**o'
    open_two2 = 'o*o*o'
    open_two3 = 'o*oo*o'

    closed_two1 = 'x**o'
    closed_two2 = 'x*o*o'
    closed_two3 = 'o*o*x'
    closed_two4 = 'o**x'

    five_count = line.count(five)
    four_count = line.count(open_four)
    cfour_count = line.count(closed_four1) + line.count(closed_four2) + line.count(closed_four3) + line.count(
        closed_four4) + line.count(closed_four5)
    three_count = line.count(open_three1) + line.count(open_three2) + line.count(open_three3) + line.count(open_three4)
    cthree_count = line.count(closed_three1) + line.count(closed_three2) + line.count(closed_three3) + line.count(
        closed_three4) + line.count(closed_three5) + line.count(closed_three6)
    two_count = line.count(open_two1) + line.count(open_two2) + line.count(open_two3)
    ctwo_count = line.count(closed_two1) + line.count(closed_two2) + line.count(closed_two3) + line.count(closed_two4)

    if five_count:
        return 1000000
    if four_count:
        return 99999
    if cfour_count + three_count > 1:
        return 9000
    score = 200 * (cfour_count + three_count) + 10 * (cthree_count + two_count) + 5 * (ctwo_count)

    return score


def get_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                score = 9 - max(abs(i - 9), abs(j - 9))
                moves.append((score, i, j))
    moves.sort()
    moves.reverse()
    return moves
