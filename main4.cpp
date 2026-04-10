#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <queue>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define MAX_ROWS 20
#define MAX_COLS 20
constexpr int SHAPE_SIZE = 4;

constexpr int DEPTH_LIMIT = 4;

enum Type { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

// Communication Tags
constexpr int TAG_REQUEST_WORK = 1;
constexpr int TAG_TASK         = 2;
constexpr int TAG_NEW_MIN      = 3;
constexpr int TAG_UPDATE_MIN   = 4;
constexpr int TAG_TERMINATE    = 5;
constexpr int TAG_RESULT       = 6;

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
    void init(int R, int C) { counts[NOT_DECIDED] = R * C; }
};

struct State {
    Solution sol;
    Coordinates p;
};

class Solver {
public:
    Solver(int rank, int size) : mpi_rank(rank), mpi_size(size) {}

    bool read() {
        if (mpi_rank != 0) return true; // Only Master reads
        if (!(cin >> R >> C)) return false;

        vector<int> allPrices(R * C);
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
        for (int i = 0; i < k; i++) trivialBound += allPrices[i];
        
        return true;
    }

    void broadcastParams() {
        MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&prices, MAX_ROWS * MAX_COLS, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&trivialBound, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    void solve() {
        if (mpi_size == 1) {
            bestSolution.price = INT_MAX;
            bestPriceShared = INT_MAX;
            masterLogic();
            return;
        }

        if (mpi_rank == 0) masterLogic();
        else slaveLogic();
    }

private:
    int mpi_rank, mpi_size;
    int R = 0, C = 0;
    int trivialBound = 0;
    int prices[MAX_ROWS][MAX_COLS];
    
    Solution bestSolution;
    int bestPriceShared = INT_MAX;
    
    void masterLogic() {
        bestPriceShared = INT_MAX;
        bestSolution.price = INT_MAX;

        Solution initialSolution;
        initialSolution.init(R, C);
        
        queue<State> q;
        q.push({initialSolution, {0, 0}});
        vector<State> work;
        const size_t ENOUGH_STATES = 30;

        // BFS Generation
        while (!q.empty() && q.size() + work.size() < ENOUGH_STATES) {
            State current = q.front();
            q.pop();
            generateNextStatesBFS(current, q, work);
        }
        while (!q.empty()) {
            work.push_back(q.front());
            q.pop();
        }

        reverse(work.begin(), work.end());

        int active_slaves = mpi_size - 1;
        
        while (active_slaves > 0) {
            // cout << active_slaves << endl;
            // cout << "work remaining: " <<  work.size() << endl;
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            if (tag == TAG_REQUEST_WORK) {
                MPI_Recv(nullptr, 0, MPI_BYTE, src, TAG_REQUEST_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (work.empty() || bestPriceShared == trivialBound) {
                    MPI_Send(nullptr, 0, MPI_BYTE, src, TAG_TERMINATE, MPI_COMM_WORLD);
                    // cout << "sending terminate" << endl;
                } else {

                    
                    while (!work.empty() && work.back().sol.price >= bestPriceShared) {work.pop_back();}
                    
                    if(work.empty()){

                        MPI_Send(nullptr, 0, MPI_BYTE, src, TAG_TERMINATE, MPI_COMM_WORLD);
                    } else {
                        State task = work.back(); work.pop_back();
                    
                        MPI_Send(&task, sizeof(State), MPI_BYTE, src, TAG_TASK, MPI_COMM_WORLD);
                    }

                }
            } else if (tag == TAG_NEW_MIN) {
                int reported_min;
                MPI_Recv(&reported_min, 1, MPI_INT, src, TAG_NEW_MIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (reported_min < bestPriceShared) {
                    bestPriceShared = reported_min;
                    // cout << bestPriceShared << endl;
                    // Rozeslání nového minima přes blokující Send
                    for (int i = 1; i < mpi_size; i++) {
                        if (i != src) {
                            MPI_Send(&bestPriceShared, 1, MPI_INT, i, TAG_UPDATE_MIN, MPI_COMM_WORLD);
                        }
                    }
                }
            } else if(tag == TAG_RESULT) {
                active_slaves--;
                Solution sol;
                MPI_Recv(&sol, sizeof(Solution), MPI_BYTE, src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // cout << "received solution " << sol.price << endl;
                if (sol.price <= bestSolution.price) {
                    bestSolution = sol;
                }
            } else {
                cout << "Received unexpected message from a slave!" << endl;
                exit(1);
            }
        }

        print();
    }

    void slaveLogic() {
        bestPriceShared = INT_MAX;
        bestSolution.price = INT_MAX;

        while (true) {
            MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_REQUEST_WORK, MPI_COMM_WORLD);
            
            bool waiting_for_task = true;
            while (waiting_for_task) {
                MPI_Status status;
                MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == TAG_TERMINATE) {
                    MPI_Recv(nullptr, 0, MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(&bestSolution, sizeof(Solution), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
                    // cout << "Sending solution " << bestSolution.price  << " at best shared min: " << bestPriceShared << endl;
                    return; 
                }
                else if (status.MPI_TAG == TAG_TASK) {
                    State task;
                    MPI_Recv(&task, sizeof(State), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    slaveSolveParalell(task.p, task.sol);
                    cout << "solved " << bestPriceShared << endl;
                    waiting_for_task = false; 
                }
                else if (status.MPI_TAG == TAG_UPDATE_MIN) {
                    int new_min;
                    MPI_Recv(&new_min, 1, MPI_INT, 0, TAG_UPDATE_MIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            

                    if (new_min < bestPriceShared) {
                        bestPriceShared = new_min;
                    }
                }
            }
        }
        
    }



    void slaveSolveParalell(Coordinates p, Solution current) {

        #pragma omp parallel
        {
            #pragma omp single
            {
                solveRecursive(p, current, 0);
            }
        }
    }


    void solveRecursive(Coordinates p, Solution& current, int depth) {

        int currentBest;
        #pragma omp atomic read
        currentBest = bestPriceShared;

        thread_local int calls = 0;
        if ((++calls % 512) == 0) {
            int flag = 0;
            int new_min = INT_MAX;
            #pragma omp critical (mpi_comm)
            {
                MPI_Iprobe(0, TAG_UPDATE_MIN, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                if (flag) {
                    MPI_Recv(&new_min, 1, MPI_INT, 0, TAG_UPDATE_MIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            if (new_min < currentBest) {
                #pragma omp critical(update_best)
                {
                    if (new_min < bestPriceShared) {
                        #pragma omp atomic write
                        bestPriceShared = new_min;
                    }
                }
            }
        }

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
                            MPI_Send(&bestPriceShared, 1, MPI_INT, 0, TAG_NEW_MIN, MPI_COMM_WORLD);
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
                if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                    Solution nextState = current; 
                    putShape(nextState, shape, p.r, p.c);
                    
                    #pragma omp task shared(bestSolution, bestPriceShared) firstprivate(nextState)
                    {
                        solveRecursive(p.next(C), nextState, depth + 1);
                    }
                }
            }

            Solution nextStateClear = current;
            nextStateClear.cellType[p.r][p.c] = CLEAR;
            bool valid = true;
            for (const auto& shape : ShapesLowerRight) {
                if (canPutShape(nextStateClear, shape, p.r, p.c, CLEAR)) {
                    valid = false; break;
                }
            }
            
            if (valid) {
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
                if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                    putShape(current, shape, p.r, p.c);
                    solveRecursive(p.next(C), current, depth + 1);
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
                solveRecursive(p.next(C), current, depth + 1);
                current.counts[NOT_DECIDED]++;
                current.counts[CLEAR]--;
                current.price -= prices[p.r][p.c];
            }
            current.cellType[p.r][p.c] = NOT_DECIDED;
        }
    }


    void slaveSolveParalellOld(Coordinates p, Solution current) {        
        // Sequence part - generating states using BFS
        queue<State> q;
        q.push({current, p});
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
        // cout << "solved" << endl;
    }

    void solveAlmostSeq(Coordinates p, Solution current) {
        int currentBest;
        #pragma omp atomic read
        currentBest = bestPriceShared;


        // thread_local int calls = 0;
        // if ((++calls % 64) == 0) {
        //     int flag = 0;
        //     int new_min = INT_MAX;
        //     #pragma omp critical (mpi_comm)
        //     {
        //         MPI_Iprobe(0, TAG_UPDATE_MIN, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        //         if (flag) {
        //             MPI_Recv(&new_min, 1, MPI_INT, 0, TAG_UPDATE_MIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //         }
        //     }

        //     if (new_min < currentBest) {
        //         #pragma omp critical(update_best)
        //         {
        //             if (new_min < bestPriceShared) {
        //                 #pragma omp atomic write
        //                 bestPriceShared = new_min;
        //             }
        //         }
        //     }
        // }

        if (current.price >= currentBest) return;
        if (currentBest == trivialBound) return;
        if (abs(current.counts[Z] - current.counts[T]) - 1 > current.counts[NOT_DECIDED] / 4) return;

        if (p.r >= R) {
            if (abs(current.counts[Z] - current.counts[T]) <= 1) {
                if (current.price < currentBest) {
                    #pragma omp critical(update_best)
                    {
                        if (current.price < bestPriceShared) {
                            #pragma omp atomic write
                            bestPriceShared = current.price;
                            bestSolution = current;
                            // MPI_Send(&bestPriceShared, 1, MPI_INT, 0, TAG_NEW_MIN, MPI_COMM_WORLD);
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
};

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        cerr << "Error: MPI does not support multithreading." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Solver solver(rank, size);
    
    if (solver.read()) {
        solver.broadcastParams();

        // Synchronizace a začátek měření
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        solver.solve();

        // Synchronizace a konec měření
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        // Výpis času pouze z hlavního vlákna
        if (rank == 0) {
            cout << "Cas behu: " << (end_time - start_time) << " sekund" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}