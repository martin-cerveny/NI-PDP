#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <omp.h>

using namespace std;

constexpr int SHAPE_SIZE = 4;
constexpr int DEPTH_LIMIT = 2;

enum Type { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

class Coordinates {
public:
    int r, c;
    
    [[nodiscard]] Coordinates next(const int cols) const {
        if (c+1 >= cols) {
            return {r + 1, 0};
        }
        return {r, c + 1};
    }
};

class Shape {
public:
    Type type;
    Coordinates tiles[SHAPE_SIZE];
};

constexpr Shape ShapesUpperLeft[] = {
    {T, {{0, 0}, {1, -1}, {1, 0}, {1, 1}}},
    {T, {{0, 0}, {1, -1}, {1, 0}, {2, 0}}},
    {T, {{0, 0}, {0, 1}, {0, 2}, {1, 1}}},
    {T, {{0, 0}, {1, 0}, {1, 1}, {2, 0}}},

    {Z, {{0,0}, {0,1}, {1, -1}, {1, 0}}},
    {Z, {{0,0}, {0, 1}, {1, 1}, {1, 2}}},
    {Z, {{0,0}, {1,-1}, {1, 0}, {2, -1}}},
    {Z, {{0,0}, {1,0}, {1, 1}, {2, 1}}},
};

constexpr Shape ShapesLowerRight[] = {
    {Z, {{-1, -2}, {-1, -1}, {0, -1}, {0, 0}}},
    {Z, {{-1, 0}, {0, -1}, {0, 0}, {1, -1}}},
    {Z, {{0, -1}, {0, 0}, {1, -2}, {1, -1}}},
    {Z, {{-2, -1}, {-1, -1}, {-1, 0}, {0, 0}}},

    {T, {{0, -2}, {0, -1}, {0, 0}, {1, -1}}},
    {T, {{-2, 0}, {-1, -1}, {-1, 0}, {0, 0}}},
    {T, {{-1, -1}, {0, -2}, {0, -1}, {0, 0}}},
    {T, {{-1, -1}, {0, -1}, {0, 0}, {1, -1}}}
};

class Solution {
public:
    int price;
    vector<vector<int>> shapeID;
    vector<vector<Type>> cellType;
    int counts[COUNT_OF_TYPES]{};

    Solution(const int R, const int C)
    : price(0), shapeID(R, vector<int>(C, 0)), cellType(R, vector<Type>(C, NOT_DECIDED)) {
        for (auto& count: counts) {
            count = 0;
        }
        counts[NOT_DECIDED] = R*C;
    }
};

class Solver{
public:
    bool read() {
        if (!(cin >> R >> C)) return false;
        prices = vector<vector<int>>(R, vector<int>(C));
        vector<int> allPrices(R*C);
        for (int r = 0, idx = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (!(cin >> prices[r][c])) return false;
                allPrices[idx] = prices[r][c];
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
        bestSolution = Solution(R, C);
        bestSolution.price = INT_MAX;
        bestPriceShared = INT_MAX;
        
        Solution initialSolution = Solution(R, C);
        initialSolution.price = 0;

        #pragma omp parallel
        {
            #pragma omp single
            {
                solveRecursive({0, 0}, initialSolution, 0);
            }
        }
    }

    void print() const {
         for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                switch (bestSolution.cellType[r][c]) {
                    case CLEAR: cout << prices[r][c]; break;
                    case Z: cout << 'Z' << bestSolution.shapeID[r][c]; break;
                    case T: cout << 'T' << bestSolution.shapeID[r][c]; break;
                    default: cout << '?'; break;
                }
                cout << '\t';
            }
            cout << endl;
        }
        cout << bestSolution.price << endl;
    }

private:
    int R = 0;
    int C = 0;
    int trivialBound = 0;
    vector<vector<int>> prices;
    Solution bestSolution = Solution(0,0);
    int bestPriceShared = INT_MAX;
    
    void putShapeState(Solution& state, const Shape& shape, const int r, const int c) const {
        state.counts[shape.type]++;
        state.counts[NOT_DECIDED] -= SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            state.cellType[rTile][cTile] = shape.type;
            state.shapeID[rTile][cTile] = state.counts[shape.type];
        }
    }
    
    void clearShapeState(Solution& state, const Shape& shape, const int r, const int c) const {
        state.counts[shape.type]--;
        state.counts[NOT_DECIDED] += SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            state.cellType[rTile][cTile] = NOT_DECIDED;
        }
    }

    [[nodiscard]] bool canPutShapeState(const Solution& state, const Shape& shape, const int r, const int c, const Type allowed) const {
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            if (rTile < 0 || rTile >= R || cTile < 0 || cTile >= C) {
                return false;
            }
            if (state.cellType[rTile][cTile] != allowed) {
                return false;
            }
        }
        return true;
    }

    void solveRecursive(Coordinates p, Solution& current, int depth) {
        int currentBest;
        #pragma omp atomic read
        currentBest = bestPriceShared;

        if (current.price >= currentBest) return;
        if (currentBest == trivialBound) return;
        if (abs(current.counts[Z] - current.counts[T]) - 1 > current.counts[NOT_DECIDED] / 4) return;

        if (p.r >= R) {
            if (abs(current.counts[Z] - current.counts[T]) <= 1) {
                if (current.price < currentBest) {
                    #pragma omp critical
                    {
                        if (current.price < bestPriceShared) {
                            #pragma omp atomic write
                            bestPriceShared = current.price;
        
                            bestSolution = current;
                        }
                    }
                }
            }
            return;
        }

        if (current.cellType[p.r][p.c] != NOT_DECIDED) {
            solveRecursive(p.next(C), current, depth);
            return;
        }

        if (depth < DEPTH_LIMIT) {
            for (const auto& shape : ShapesUpperLeft) {
                if (canPutShapeState(current, shape, p.r, p.c, NOT_DECIDED)) {
                    Solution nextState = current; 
                    putShapeState(nextState, shape, p.r, p.c);
                    
                    #pragma omp task shared(bestSolution, bestPriceShared) firstprivate(nextState)
                    {
                        solveRecursive(p.next(C), nextState, depth + 1);
                    }
                }
            }

            Solution nextStateClear = current;
            nextStateClear.cellType[p.r][p.c] = CLEAR;
            bool valid = false;
            for (const auto& shape : ShapesLowerRight) {
                if (canPutShapeState(nextStateClear, shape, p.r, p.c, CLEAR)) {
                    valid = true; break;
                }
            }
            
            if (!valid) {
                nextStateClear.counts[NOT_DECIDED]--;
                nextStateClear.counts[CLEAR]++;
                nextStateClear.price += prices[p.r][p.c];
                
                #pragma omp task shared(bestSolution, bestPriceShared) firstprivate(nextStateClear)
                {
                    solveRecursive(p.next(C), nextStateClear, depth + 1);
                }
            }
            #pragma omp taskwait 
        } else {
            for (const auto& shape : ShapesUpperLeft) {
                if (canPutShapeState(current, shape, p.r, p.c, NOT_DECIDED)) {
                    putShapeState(current, shape, p.r, p.c);
                    solveRecursive(p.next(C), current, depth + 1);
                    clearShapeState(current, shape, p.r, p.c);
                }
            }

            current.cellType[p.r][p.c] = CLEAR;
            bool valid = false;
            for (const auto& shape : ShapesLowerRight) {
                if (canPutShapeState(current, shape, p.r, p.c, CLEAR)) {
                    valid = true; break;
                }
            }
            if (!valid) {
                current.counts[NOT_DECIDED]--;
                current.counts[CLEAR]++;
                current.price += prices[p.r][p.c];
                solveRecursive(p.next(C), current, depth + 1);
                current.counts[NOT_DECIDED]++;
                current.counts[CLEAR]--;
                current.price -= prices[p.r][p.c];
            }
            current.cellType[p.r][p.c] = NOT_DECIDED;
        }
    }
};

int main() {
    Solver solver;
    solver.read();
    solver.solve();
    solver.print();
    return 0;
}