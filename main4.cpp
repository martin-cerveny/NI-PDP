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

enum Type { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

// MPI TAGY
#define TAG_WORK 1
#define TAG_RESULT 2
#define TAG_NEW_BOUND 3
#define TAG_KILL 4

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

struct WorkResult {
    int best_price;
    Solution best_sol;
};

class Solver {
public:
    int mpi_rank, mpi_size;

    Solver(int rank, int size) : mpi_rank(rank), mpi_size(size) {}

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

    void run() {
        if (mpi_rank == 0) {
            masterRoutine();
        } else {
            slaveRoutine();
        }
    }

private:
    int R = 0;
    int C = 0;
    int trivialBound = 0;
    int prices[MAX_ROWS][MAX_COLS];
    
    Solution bestSolution;
    int bestPriceShared = INT_MAX;
    const int DEPTH_LIMIT = 2; // For task parallelism

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

    void checkMPIUpdates() {
    thread_local int poll_counter = 0;
    
    if (++poll_counter < 10000) return; 
    poll_counter = 0;

    int flag;
    MPI_Iprobe(0, TAG_NEW_BOUND, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        int new_bound;
        MPI_Recv(&new_bound, 1, MPI_INT, 0, TAG_NEW_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #pragma omp critical
        {
            if (new_bound < bestPriceShared) bestPriceShared = new_bound;
        }
    }
}

    void broadcastNewBoundSlave(int new_val) {
        MPI_Request req;
        MPI_Isend(&new_val, 1, MPI_INT, 0, TAG_NEW_BOUND, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }

    // Solvers
    void solveAlmostSeq(Coordinates p, Solution current) {
        checkMPIUpdates(); // MPI Polling

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
                            bestPriceShared = current.price;
                            bestSolution = current;
                            broadcastNewBoundSlave(current.price);
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

    void solveRecursiveTask(Coordinates p, Solution& current, int depth) {
        checkMPIUpdates(); // MPI Polling

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
                            bestPriceShared = current.price;
                            bestSolution = current;
                            broadcastNewBoundSlave(current.price);
                        }
                    }
                }
            }
            return;
        }

        if (current.cellType[p.r][p.c] != NOT_DECIDED) {
            solveRecursiveTask(p.next(C), current, depth);
            return;
        }

        if (depth < DEPTH_LIMIT) {
            for (const auto& shape : ShapesUpperLeft) {
                if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                    Solution nextState = current; 
                    putShape(nextState, shape, p.r, p.c);
                    
                    #pragma omp task shared(bestPriceShared) firstprivate(nextState)
                    {
                        solveRecursiveTask(p.next(C), nextState, depth + 1);
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
                
                #pragma omp task shared(bestPriceShared) firstprivate(nextStateClear)
                {
                    solveRecursiveTask(p.next(C), nextStateClear, depth + 1);
                }
            }
            #pragma omp taskwait 
        } else {
            // Fallback na sekvenční s OMP prořezáváním
            solveAlmostSeq(p, current); 
        }
    }

    void generateNextStatesBFS(State current, queue<State>& q, vector<State>& items, size_t limit) {
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

    void masterRoutine() {
        bestSolution.price = INT_MAX;
        bestPriceShared = INT_MAX;

        Solution initialSolution;
        initialSolution.init(R, C);
        
        queue<State> q;
        q.push({initialSolution, {0, 0}});
        vector<State> work_items;
        
        const size_t ENOUGH_STATES = mpi_size * 100;

        while (!q.empty() && q.size() + work_items.size() < ENOUGH_STATES) {
            State current = q.front();
            q.pop();
            generateNextStatesBFS(current, q, work_items, ENOUGH_STATES);
        }
        while (!q.empty()) {
            work_items.push_back(q.front());
            q.pop();
        }

        queue<int> idle_slaves;
        for (int i = 1; i < mpi_size; i++) idle_slaves.push(i);
        int active_workers = 0;
        size_t tasks_sent = 0;

        while (tasks_sent < work_items.size() || active_workers > 0) {
            while (tasks_sent < work_items.size() && !idle_slaves.empty()) {
                int dest = idle_slaves.front();
                idle_slaves.pop();
                
                MPI_Send(&work_items[tasks_sent], sizeof(State), MPI_BYTE, dest, TAG_WORK, MPI_COMM_WORLD);
                MPI_Send(&bestPriceShared, 1, MPI_INT, dest, TAG_NEW_BOUND, MPI_COMM_WORLD);
                tasks_sent++;
                active_workers++;
            }

            if (active_workers > 0) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == TAG_RESULT) {
                    WorkResult res;
                    MPI_Recv(&res, sizeof(WorkResult), MPI_BYTE, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    if (res.best_price < bestPriceShared) {
                        bestPriceShared = res.best_price;
                        bestSolution = res.best_sol;
                    }
                    active_workers--;
                    idle_slaves.push(status.MPI_SOURCE);
                    
                } else if (status.MPI_TAG == TAG_NEW_BOUND) {
                    int new_bound;
                    MPI_Recv(&new_bound, 1, MPI_INT, status.MPI_SOURCE, TAG_NEW_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    if (new_bound < bestPriceShared) {
                        bestPriceShared = new_bound;
                        for (int i = 1; i < mpi_size; i++) {
                            if (i != status.MPI_SOURCE) {
                                MPI_Request req;
                                MPI_Isend(&bestPriceShared, 1, MPI_INT, i, TAG_NEW_BOUND, MPI_COMM_WORLD, &req);
                                MPI_Request_free(&req);
                            }
                        }
                    }
                }
            }
        }

        for (int i = 1; i < mpi_size; i++) {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_KILL, MPI_COMM_WORLD);
        }
        print();
    }

    void slaveRoutine() {
        bool konec = false;
        
        const bool USE_TASK_PARALLELISM = true; 

        while (!konec) {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_KILL) {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_KILL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                konec = true;
                
            } else if (status.MPI_TAG == TAG_NEW_BOUND) {
                int b;
                MPI_Recv(&b, 1, MPI_INT, 0, TAG_NEW_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (b < bestPriceShared) bestPriceShared = b;
                
            } else if (status.MPI_TAG == TAG_WORK) {
                State s;
                MPI_Recv(&s, sizeof(State), MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&bestPriceShared, 1, MPI_INT, 0, TAG_NEW_BOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                bestSolution.price = INT_MAX;

                if (USE_TASK_PARALLELISM) {
                    #pragma omp parallel shared(bestPriceShared)
                    {
                        #pragma omp single
                        {
                            solveRecursiveTask(s.p, s.sol, 0);
                        }
                    }
                } else {
                    queue<State> local_q;
                    local_q.push(s);
                    vector<State> local_items;
                    
                    while (!local_q.empty() && local_q.size() + local_items.size() < 500) {
                        State curr = local_q.front();
                        local_q.pop();
                        generateNextStatesBFS(curr, local_q, local_items, 500);
                    }
                    while (!local_q.empty()) {
                        local_items.push_back(local_q.front());
                        local_q.pop();
                    }

                    int is = local_items.size();
                    #pragma omp parallel for schedule(dynamic) shared(bestPriceShared)
                    for (int i = 0; i < is; i++) {
                        solveAlmostSeq(local_items[i].p, local_items[i].sol);
                    }
                }

                WorkResult res;
                res.best_price = bestSolution.price;
                res.best_sol = bestSolution;
                MPI_Send(&res, sizeof(WorkResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
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
};

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "FATAL: MPI implementation does not support multithreading." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Solver solver(rank, size);
    
   
    bool ok = false;
    if (rank == 0) {
        ok = solver.read();
    }
    MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    
    if (ok) {
        MPI_Bcast(&solver, sizeof(Solver), MPI_BYTE, 0, MPI_COMM_WORLD);
        // Refresh rank and size after Bcast - they were overwritten
        solver.mpi_rank = rank;
        solver.mpi_size = size;
        
        solver.run();
    }

    MPI_Finalize();
    return 0;
}