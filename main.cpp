#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>

using namespace std;

constexpr int SHAPE_SIZE = 4;

enum Type { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES};

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
        currentSolution = Solution(R, C);
        bestSolution = Solution(R, C);
        bestSolution.price = INT_MAX;
        currentSolution.price = 0;
        solveRecursive({0, 0});
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
        cout << "Calls: " << totalCalls << endl;
    }
private:
    int R = 0;
    int C = 0;
    int totalCalls = 0;
    int trivialBound = 0;
    vector<vector<int>> prices;
    Solution currentSolution = Solution(0,0);
    Solution bestSolution = Solution(0,0);
    
    void putShape(const Shape& shape, const int r, const int c) {
        currentSolution.counts[shape.type]++;
        currentSolution.counts[NOT_DECIDED] -= SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            currentSolution.cellType[rTile][cTile] = shape.type;
            currentSolution.shapeID[rTile][cTile] = currentSolution.counts[shape.type];
        }
    }
    
    void clearShape(const Shape& shape, const int r, const int c) {
        currentSolution.counts[shape.type]--;
        currentSolution.counts[NOT_DECIDED] += SHAPE_SIZE;
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            currentSolution.cellType[rTile][cTile] = NOT_DECIDED;
        }
    }
    


    [[nodiscard]] bool canPutShape(const Shape& shape, const int r, const int c, const Type allowed) const {
        for (auto &[rd, cd] : shape.tiles) {
            const int rTile = r + rd;
            const int cTile = c + cd;
            if (rTile < 0 || rTile >= R || cTile < 0 || cTile >= C) {
                return false;
            }

            if (currentSolution.cellType[rTile][cTile] != allowed) {
                return false;
            }

        }

        return true;

    }

    void solveRecursive(Coordinates p) {
        totalCalls++;
        if (currentSolution.price >= bestSolution.price) return;
        if (bestSolution.price == trivialBound) return;
        if (abs(currentSolution.counts[Z] - currentSolution.counts[T]) - 1 > currentSolution.counts[NOT_DECIDED] / 4)
            return;


        if (p.r >= R) {
            if (abs(currentSolution.counts[Z] - currentSolution.counts[T]) <= 1) {
                if (currentSolution.price < bestSolution.price) {
                    bestSolution = currentSolution;
                }
            }
            return;
        }

        if (currentSolution.cellType[p.r][p.c] != NOT_DECIDED) {
            solveRecursive(p.next(C));
            return;
        }

        // 1) Try place shape
        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(shape, p.r, p.c, NOT_DECIDED)) {
                putShape(shape, p.r, p.c);
                solveRecursive(p.next(C));
                if (bestSolution.price == trivialBound) return;
                clearShape(shape, p.r, p.c);
            }
        }

        // 2) Try leave not covered
        // A decision to create CLEAR tile cannot create a space, where some shape might fit
        currentSolution.cellType[p.r][p.c] = CLEAR;
        for (const auto& shape : ShapesLowerRight) {
            if (canPutShape(shape, p.r, p.c, CLEAR)) {
                currentSolution.cellType[p.r][p.c] = NOT_DECIDED;
                return;
            }
        }
        currentSolution.counts[NOT_DECIDED] --;
        currentSolution.counts[CLEAR] ++;
        currentSolution.price += prices[p.r][p.c];
        solveRecursive(p.next(C));
        if (bestSolution.price == trivialBound) return;
        currentSolution.cellType[p.r][p.c] = NOT_DECIDED;
        currentSolution.counts[NOT_DECIDED] ++;
        currentSolution.counts[CLEAR] --;
        currentSolution.price -= prices[p.r][p.c];

    }


};


int main() {

    Solver solver;
    solver.read();
    solver.solve();
    solver.print();

    return 0;

}
