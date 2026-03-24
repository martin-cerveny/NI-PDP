#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <queue>
#include <omp.h>

using namespace std;

#define MAX_ROWS 20
#define MAX_COLS 20
constexpr int SHAPE_SIZE = 4;

enum Type { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

class Coordinates {
public:
    int r, c;
    [[nodiscard]] Coordinates next(const int cols) const {
        if (c + 1 >= cols) return {r + 1, 0};
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
    int shapeID[MAX_ROWS][MAX_COLS];
    Type cellType[MAX_ROWS][MAX_COLS];
    int counts[COUNT_OF_TYPES];

    Solution() {
        price = 0;
        for (int &count : counts) count = 0;
        for (int r = 0; r < MAX_ROWS; r++) {
            for (int c = 0; c < MAX_COLS; c++) {
                shapeID[r][c] = 0;
                cellType[r][c] = NOT_DECIDED;
            }
        }
    }
    void init(int R, int C) {
        counts[NOT_DECIDED] = R * C;
    }
};

struct State {
    Solution sol;
    Coordinates p;
};

class Solver {
public:
    bool read() {
        if (!(cin >> R >> C)) return false;

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
        bestSolution.price = INT_MAX;
        bestPriceShared = INT_MAX;

        Solution initialSolution;
        initialSolution.init(R, C);
        
        // Sequence part - generating states using BFS
        queue<State> q;
        q.push({initialSolution, {0, 0}});
        vector<State> items;
        const size_t ENOUGH_STATES = 5000;

        while (!q.empty() && q.size() + items.size() < ENOUGH_STATES) {
            State current = q.front();
            q.pop();
            generateNextStatesBFS(current, q, items);
        }
        while (!q.empty()) {
            items.push_back(q.front());
            q.pop();
        }

        int is = items.size();
        
        // Data paralelism
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < is; i++) {
            solveAlmostSeq(items[i].p, items[i].sol);
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
    int prices[MAX_ROWS][MAX_COLS];
    
    Solution bestSolution;
    int bestPriceShared = INT_MAX;

    void putShape(Solution& state, const Shape& shape, const int r, const int c) const {
        state.counts[shape.type]++;
        state.counts[NOT_DECIDED] -= SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            state.cellType[r + rd][c + cd] = shape.type;
            state.shapeID[r + rd][c + cd] = state.counts[shape.type];
        }
    }

    void clearShape(Solution& state, const Shape& shape, const int r, const int c) const {
        state.counts[shape.type]--;
        state.counts[NOT_DECIDED] += SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            state.cellType[r + rd][c + cd] = NOT_DECIDED;
        }
    }

    [[nodiscard]] bool canPutShape(const Solution& state, const Shape& shape, const int r, const int c, const Type allowed) const {
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            if (rTile < 0 || rTile >= R || cTile < 0 || cTile >= C) return false;
            if (state.cellType[rTile][cTile] != allowed) return false;
        }
        return true;
    }

    void generateNextStatesBFS(State current, queue<State>& q, vector<State>& items) {
        Coordinates p = current.p;
        while (p.r < R && current.sol.cellType[p.r][p.c] != NOT_DECIDED) {
            p = p.next(C);
        }

        if (p.r >= R) {
            current.p = p;
            items.push_back(current);
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current.sol, shape, p.r, p.c, NOT_DECIDED)) {
                State nextState = current;
                putShape(nextState.sol, shape, p.r, p.c);
                nextState.p = p.next(C);
                q.push(nextState);
            }
        }

        State nextStateClear = current;
        nextStateClear.sol.cellType[p.r][p.c] = CLEAR;
        bool valid = true;
        for (const auto& shape : ShapesLowerRight) {
            if (canPutShape(nextStateClear.sol, shape, p.r, p.c, CLEAR)) {
                valid = false; break;
            }
        }
        if (valid) {
            nextStateClear.sol.counts[NOT_DECIDED]--;
            nextStateClear.sol.counts[CLEAR]++;
            nextStateClear.sol.price += prices[p.r][p.c];
            nextStateClear.p = p.next(C);
            q.push(nextStateClear);
        }
    }

    void solveAlmostSeq(Coordinates p, Solution current) {
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
            solveAlmostSeq(p.next(C), current);
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                putShape(current, shape, p.r, p.c);
                solveAlmostSeq(p.next(C), current);
                clearShape(current, shape, p.r, p.c);
            }
        }

        current.cellType[p.r][p.c] = CLEAR;
        bool valid = true;
        for (const auto& shape : ShapesLowerRight) {
            if (canPutShape(current, shape, p.r, p.c, CLEAR)) {
                valid = false; break;
            }
        }
        if (valid) {
            current.counts[NOT_DECIDED]--;
            current.counts[CLEAR]++;
            current.price += prices[p.r][p.c];
            solveAlmostSeq(p.next(C), current);
            current.counts[NOT_DECIDED]++;
            current.counts[CLEAR]--;
            current.price -= prices[p.r][p.c];
        }
        current.cellType[p.r][p.c] = NOT_DECIDED;
    }
};

int main() {
    Solver solver;
    if (solver.read()) {
        solver.solve();
        solver.print();
    }
    return 0;
}