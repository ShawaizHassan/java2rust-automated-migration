// src/main/java/com/example/KnightsTour.java

import java.util.Optional;

public final class KnightsTour {
    private KnightsTour() {
    }

    // Fixed move order (dx, dy) — must match Rust
    private static final int[][] MOVES = new int[][] {
            { 2, 1 },
            { 1, 2 },
            { -1, 2 },
            { -2, 1 },
            { -2, -1 },
            { -1, -2 },
            { 1, -2 },
            { 2, -1 },
    };

    /**
     * Finds a knight's tour or returns Optional.empty() if none. Coordinates are
     * 0-based.
     */
    public static Optional<int[][]> findKnightTour(int sizeX, int sizeY, int startX, int startY) {
        if (sizeX <= 0 || sizeY <= 0)
            return Optional.empty();
        int[][] board = new int[sizeY][sizeX]; // board[y][x]
        if (!isFree(board, sizeX, sizeY, startX, startY))
            return Optional.empty();

        board[startY][startX] = 1;
        if (solve(board, sizeX, sizeY, startX, startY, 1)) {
            return Optional.of(copy(board));
        }
        return Optional.empty();
    }

    private static boolean solve(int[][] board, int w, int h, int x, int y, int moveCount) {
        if (moveCount == w * h)
            return true;

        for (int[] m : MOVES) {
            int nx = x + m[0], ny = y + m[1];
            if (isFree(board, w, h, nx, ny)) {
                board[ny][nx] = moveCount + 1;
                if (solve(board, w, h, nx, ny, moveCount + 1))
                    return true;
                board[ny][nx] = 0; // backtrack
            }
        }
        return false;
    }

    private static boolean isFree(int[][] board, int w, int h, int x, int y) {
        return x >= 0 && y >= 0 && x < w && y < h && board[y][x] == 0;
    }

    private static int[][] copy(int[][] a) {
        int[][] b = new int[a.length][a[0].length];
        for (int i = 0; i < a.length; i++)
            System.arraycopy(a[i], 0, b[i], 0, a[i].length);
        return b;
    }
}
