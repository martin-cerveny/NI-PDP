#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <queue>
#include <omp.h>

using namespace std;

constexpr int MAX_ROWS =  20;
constexpr int MAX_COLS = 20;
constexpr int SHAPE_SIZE = 4;
constexpr int DEPTH_LIMIT = 4;

enum CellType { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

struct  Coordinates {
    int r, c;
};

struct Shape {
    CellType type;
    Coordinates tiles[SHAPE_SIZE];
};

struct CellState {
    CellType type = NOT_DECIDED;
    int id = 0;
};

struct Solution {
    CellState boardState[MAX_ROWS][MAX_COLS]{};

    inline CellType& type(int r, int c) { return boardState[r][c].type; }
    inline CellType type(int r, int c) const { return boardState[r][c].type; }
    inline CellType& type(Coordinates p) { return boardState[p.r][p.c].type; }
    inline CellType type(Coordinates p) const { return boardState[p.r][p.c].type; }
};

class Node {
public:
    int price = 0;
    int quatrominoCntPerType[COUNT_OF_TYPES]{};
    Solution solution;

    Node(const int R, const int C) {
        quatrominoCntPerType[NOT_DECIDED] = R * C;
    }

    inline CellType type(int r, int c) const { return solution.type(r, c); }
    inline CellType type(Coordinates p) const { return solution.type(p); }
    inline void setType(Coordinates p, CellType t) { solution.type(p) = t; }
    inline void setCell(int r, int c, CellType t, int id) { solution.boardState[r][c] = {t, id}; }
};

class Solver {
public:
    bool read() {
        if (!(cin >> R >> C)) return false;

        vector<int> allPrices(R*C);
        for (int r = 0, idx = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (!(cin >> pricesAssignment[r][c])) return false;
                allPrices[idx] = pricesAssignment[r][c];
                idx++;
            }
        }
        sort(allPrices.begin(), allPrices.end());

        const int k = (R * C) % 4;
        trivialBound = 0;
        for (int i = 0; i < k; i++) {
            trivialBound += allPrices[i];
        }
        return true;
    }

    void solve() {
        bestPriceShared = INT_MAX;

        Node initial(R, C);

        #pragma omp parallel
        {
            #pragma omp single
            {
                solveRecursive({0, 0}, initial, 0);
            }
        }
    }

    void print() const {
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                switch (bestSolution.boardState[r][c].type) {
                    case CLEAR: cout << pricesAssignment[r][c]; break;
                    case Z: cout << 'Z' << bestSolution.boardState[r][c].id; break;
                    case T: cout << 'T' << bestSolution.boardState[r][c].id; break;
                    default: cout << '?'; break;
                }
                cout << '\t';
            }
            cout << endl;
        }
        cout << bestPriceShared << endl;
    }
private:
    int R = 0;
    int C = 0;
    int bestPriceShared = INT_MAX;
    int trivialBound = 0;
    int pricesAssignment[MAX_ROWS][MAX_COLS]{};
    Solution bestSolution;

    static constexpr Shape ShapesUpperLeft[] = {
        {T, {{0, 0}, {1, -1}, {1, 0}, {1, 1}}},
        {T, {{0, 0}, {1, -1}, {1, 0}, {2, 0}}},
        {T, {{0, 0}, {0, 1}, {0, 2}, {1, 1}}},
        {T, {{0, 0}, {1, 0}, {1, 1}, {2, 0}}},
        {Z, {{0,0}, {0,1}, {1, -1}, {1, 0}}},
        {Z, {{0,0}, {0, 1}, {1, 1}, {1, 2}}},
        {Z, {{0,0}, {1,-1}, {1, 0}, {2, -1}}},
        {Z, {{0,0}, {1,0}, {1, 1}, {2, 1}}},
    };

    static constexpr Shape ShapesLowerRight[] = {
        {Z, {{-1, -2}, {-1, -1}, {0, -1}, {0, 0}}},
        {Z, {{-1, 0}, {0, -1}, {0, 0}, {1, -1}}},
        {Z, {{0, -1}, {0, 0}, {1, -2}, {1, -1}}},
        {Z, {{-2, -1}, {-1, -1}, {-1, 0}, {0, 0}}},
        {T, {{0, -2}, {0, -1}, {0, 0}, {1, -1}}},
        {T, {{-2, 0}, {-1, -1}, {-1, 0}, {0, 0}}},
        {T, {{-1, -1}, {0, -2}, {0, -1}, {0, 0}}},
        {T, {{-1, -1}, {0, -1}, {0, 0}, {1, -1}}}
    };

    Coordinates nextNotDecidedPosition(Coordinates position, const Solution& solution) const {
        while (position.r < R && solution.type(position) != NOT_DECIDED) {
            position.c++;
            if (position.c >= C) {
                position.r++;
                position.c = 0;
            }
        }
        return position;
    }

    static void putShape(Node& node, const Shape& shape, const int r, const int c) {
        node.quatrominoCntPerType[shape.type]++;
        node.quatrominoCntPerType[NOT_DECIDED] -= SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            node.setCell(r + rd, c + cd, shape.type, node.quatrominoCntPerType[shape.type]);
        }
    }

    static void removeShape(Node& node, const Shape& shape, const int r, const int c) {
        node.quatrominoCntPerType[shape.type]--;
        node.quatrominoCntPerType[NOT_DECIDED] += SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            node.solution.type(r + rd, c + cd) = NOT_DECIDED;
        }
    }

    bool canPutShape(const Node& node, const Shape& shape, const int r, const int c, const CellType allowed) const {
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            if (rTile < 0 || rTile >= R || cTile < 0 || cTile >= C) return false;
            if (node.type(rTile, cTile) != allowed) return false;
        }
        return true;
    }

    bool tryPutClear(Node& node, Coordinates p) const {
        node.setType(p, CLEAR);

        for (const auto& shape : ShapesLowerRight) {
            if (canPutShape(node, shape, p.r, p.c, CLEAR)) {
                node.setType(p, NOT_DECIDED);
                return false;
            }
        }

        node.quatrominoCntPerType[NOT_DECIDED]--;
        node.quatrominoCntPerType[CLEAR]++;
        node.price += pricesAssignment[p.r][p.c];
        return true;
    }

    void removeClear(Node& node, Coordinates p) const {
        node.setType(p, NOT_DECIDED);
        node.quatrominoCntPerType[NOT_DECIDED]++;
        node.quatrominoCntPerType[CLEAR]--;
        node.price -= pricesAssignment[p.r][p.c];
    }

    void solveRecursive(Coordinates p, Node& current, int depth) {
        int currentBest;
        #pragma omp atomic read relaxed
        currentBest = bestPriceShared;

        if (current.price >= currentBest || currentBest == trivialBound) return;
        if (abs(current.quatrominoCntPerType[Z] - current.quatrominoCntPerType[T]) - 1 > current.quatrominoCntPerType[NOT_DECIDED] / 4) return;

        if (p.r >= R) {
            if (abs(current.quatrominoCntPerType[Z] - current.quatrominoCntPerType[T]) <= 1) {
                if (current.price < currentBest) {
                    #pragma omp critical
                    {
                        if (current.price < bestPriceShared) {
                            #pragma omp atomic write
                            bestPriceShared = current.price;
                            bestSolution = current.solution;
                        }
                    }
                }
            }
            return;
        }

        if (depth < DEPTH_LIMIT) {
            for (const auto& shape : ShapesUpperLeft) {
                if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                    Node nextNode = current;
                    putShape(nextNode, shape, p.r, p.c);

                    #pragma omp task shared(bestSolution, bestPriceShared) firstprivate(nextNode)
                    solveRecursive(nextNotDecidedPosition(p, nextNode.solution), nextNode, depth + 1);
                }
            }

            Node nextNodeClear = current;
            if (tryPutClear(nextNodeClear, p)) {
                #pragma omp task shared(bestSolution, bestPriceShared) firstprivate(nextNodeClear)
                solveRecursive(nextNotDecidedPosition(p, nextNodeClear.solution), nextNodeClear, depth + 1);
            }

        } else {
            for (const auto& shape : ShapesUpperLeft) {
                if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                    putShape(current, shape, p.r, p.c);
                    solveRecursive(nextNotDecidedPosition(p, current.solution), current, depth + 1);
                    removeShape(current, shape, p.r, p.c);
                }
            }

            if (tryPutClear(current, p)) {
                solveRecursive(nextNotDecidedPosition(p, current.solution), current, depth + 1);
                removeClear(current, p);
            }
        }
    }
};

int main() {
    Solver solver;
    if (solver.read()) {
        const double start_time = omp_get_wtime();

        solver.solve();


        const double end_time = omp_get_wtime();

        solver.print();

        cout << "Cas behu: " << (end_time - start_time) << " sekund" << endl;
    }
    return 0;
}