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

class TilePosition {
public:
    int r, c;
};

class Shape {
public:
    Type type;
    TilePosition tiles[SHAPE_SIZE];
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
    int counts[COUNT_OF_TYPES];
    Type cellType[MAX_ROWS][MAX_COLS];


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
    void init(const int R, const int C) {
        counts[NOT_DECIDED] = R * C;
    }
};

struct State {
    Solution solution;
    TilePosition p;
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
            int currentBestLoop;
            #pragma omp atomic read relaxed
            currentBestLoop = bestPriceShared;

            if (currentBestLoop == trivialBound) continue;
            Solution localSol = items[i].solution;
            solveAlmostSeq(items[i].p, localSol);
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


    TilePosition nextPosition(const TilePosition& p) const {
        if (p.c + 1 >= C) return {p.r + 1, 0};
        return {p.r, p.c + 1};
    }

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
        TilePosition p = current.p;
        while (p.r < R && current.solution.cellType[p.r][p.c] != NOT_DECIDED) {
            p = nextPosition(p);
        }

        if (p.r >= R) {
            current.p = p;
            items.push_back(current);
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current.solution, shape, p.r, p.c, NOT_DECIDED)) {
                State nextState = current;
                putShape(nextState.solution, shape, p.r, p.c);
                nextState.p = nextPosition(p);
                q.push(nextState);
            }
        }

        State nextStateClear = current;
        nextStateClear.solution.cellType[p.r][p.c] = CLEAR;
        bool valid = true;
        for (const auto& shape : ShapesLowerRight) {
            if (canPutShape(nextStateClear.solution, shape, p.r, p.c, CLEAR)) {
                valid = false; break;
            }
        }
        if (valid) {
            nextStateClear.solution.counts[NOT_DECIDED]--;
            nextStateClear.solution.counts[CLEAR]++;
            nextStateClear.solution.price += prices[p.r][p.c];
            nextStateClear.p = nextPosition(p);
            q.push(nextStateClear);
        }
    }

    void solveAlmostSeq(TilePosition p, Solution& current) {
        int currentBest;
        #pragma omp atomic read relaxed
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
            solveAlmostSeq(nextPosition(p), current);
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                putShape(current, shape, p.r, p.c);
                solveAlmostSeq(nextPosition(p), current);
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
            solveAlmostSeq(nextPosition(p), current);
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