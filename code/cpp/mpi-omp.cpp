#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <queue>
#include <omp.h>
#include <iomanip>
#include <mpi.h>

using namespace std;

//#define ENABLE_MPI_LOGS

#ifdef ENABLE_MPI_LOGS
#define LOG_MPI(msg) do { \
std::cout << std::fixed << std::setprecision(6) << MPI_Wtime() << "s " << msg << std::endl; \
} while(false)
#else
#define LOG_MPI(msg) do {} while(false)
#endif

constexpr int MAX_ROWS = 20;
constexpr int MAX_COLS = 20;
constexpr int SHAPE_SIZE = 4;
constexpr int ENOUGH_STATES_COEF_MASTER = 10;
constexpr int ENOUGH_STATES_COEF_SLAVE = 500;

enum CellType { Z, T, CLEAR, NOT_DECIDED, COUNT_OF_TYPES };

// Communication Tags
constexpr int TAG_REQUEST_WORK = 1;
constexpr int TAG_TASK         = 2;
constexpr int TAG_NEW_MIN      = 3;
constexpr int TAG_UPDATE_MIN   = 4;
constexpr int TAG_TERMINATE    = 5;
constexpr int TAG_RESULT       = 6;

struct Coordinates {
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

    Node() {}
    Node(const int R, const int C) {
        quatrominoCntPerType[NOT_DECIDED] = R * C;
    }

    inline CellType type(int r, int c) const { return solution.type(r, c); }
    inline CellType type(Coordinates p) const { return solution.type(p); }
    inline void setType(Coordinates p, CellType t) { solution.type(p) = t; }
    inline void setCell(int r, int c, CellType t, int id) { solution.boardState[r][c] = {t, id}; }
};

struct WorkData {
    Node node;
    Coordinates p;
};

struct ResultPacket {
    int price;
    Solution sol;
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
                if (!(cin >> pricesAssignment[r][c])) return false;
                allPrices[idx] = pricesAssignment[r][c];
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
        MPI_Bcast(&pricesAssignment, MAX_ROWS * MAX_COLS, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&trivialBound, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    void solve(const int masterEnoughStates, const int slaveEnoughStates) {
        if (mpi_size == 1) {
            bestPriceShared = INT_MAX;
            Node initialSolution(R, C);
            processTaskParallel({initialSolution, {0, 0}}, slaveEnoughStates);
            print();
            return;
        }

        if (mpi_rank == 0) masterLogic(masterEnoughStates);
        else slaveLogic(slaveEnoughStates);
    }

private:
    int mpi_rank, mpi_size;
    int R = 0, C = 0;
    int trivialBound = 0;
    int pricesAssignment[MAX_ROWS][MAX_COLS]{};

    int bestPriceShared = INT_MAX;
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

    void masterLogic(const int masterEnoughStates) {
        LOG_MPI("Number of active slaves: " << (mpi_size - 1));
        bestPriceShared = INT_MAX;
        Node initialSolution(R, C);

        queue<WorkData> q;
        q.push({initialSolution, {0, 0}});
        vector<WorkData> work;

        // BFS Generation for Master
        while (!q.empty() && q.size() + work.size() < masterEnoughStates) {
            auto [node, p] = q.front();
            q.pop();
            generateNextStatesBFS(node, p, q, work);
        }
        while (!q.empty()) {
            work.push_back(q.front());
            q.pop();
        }

        reverse(work.begin(), work.end());

        LOG_MPI("Master generated " << work.size() << " states.");

        int active_slaves = mpi_size - 1;

        while (active_slaves > 0) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            if (tag == TAG_REQUEST_WORK) {
                MPI_Recv(nullptr, 0, MPI_BYTE, src, TAG_REQUEST_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (work.empty() || bestPriceShared == trivialBound) {
                    LOG_MPI("Master sends TERMINATE to slave " << src << ".");
                    MPI_Send(nullptr, 0, MPI_BYTE, src, TAG_TERMINATE, MPI_COMM_WORLD);
                } else {
                    while (!work.empty() && work.back().node.price >= bestPriceShared) { work.pop_back(); }

                    if(work.empty()){
                        LOG_MPI("Master sends TERMINATE to slave " << src << ".");
                        MPI_Send(nullptr, 0, MPI_BYTE, src, TAG_TERMINATE, MPI_COMM_WORLD);
                    } else {
                        WorkData task = work.back();
                        work.pop_back();
                        LOG_MPI("Master sends work to slave " << src << ".");
                        MPI_Send(&task, sizeof(WorkData), MPI_BYTE, src, TAG_TASK, MPI_COMM_WORLD);
                    }
                }
            } else if (tag == TAG_NEW_MIN) {
                int reported_min;
                MPI_Recv(&reported_min, 1, MPI_INT, src, TAG_NEW_MIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (reported_min < bestPriceShared) {
                    bestPriceShared = reported_min;
                    for (int i = 1; i < mpi_size; i++) {
                        if (i != src) {
                            MPI_Send(&bestPriceShared, 1, MPI_INT, i, TAG_UPDATE_MIN, MPI_COMM_WORLD);
                        }
                    }
                }
            } else if(tag == TAG_RESULT) {
                active_slaves--;
                ResultPacket res;
                MPI_Recv(&res, sizeof(ResultPacket), MPI_BYTE, src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (res.price <= bestPriceShared) {
                    bestPriceShared = res.price;
                    bestSolution = res.sol;
                }
            } else {
                cout << "Received unexpected message from a slave!" << endl;
                exit(1);
            }
        }

        print();
    }

    void slaveLogic(const int slaveEnoughStates) {
        LOG_MPI("Slave " << mpi_rank << " ready. Threads per process: " << omp_get_max_threads());
        bestPriceShared = INT_MAX;
        while (true) {
            LOG_MPI("Slave " << mpi_rank << " requests a Task from Master.");
            MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_REQUEST_WORK, MPI_COMM_WORLD);

            bool waiting_for_task = true;
            while (waiting_for_task) {
                MPI_Status status;
                MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == TAG_TERMINATE) {
                    MPI_Recv(nullptr, 0, MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    ResultPacket res = { bestPriceShared, bestSolution };
                    MPI_Send(&res, sizeof(ResultPacket), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
                    LOG_MPI("Slave " << mpi_rank << " terminates.");
                    return;
                }
                else if (status.MPI_TAG == TAG_TASK) {
                    WorkData task;
                    MPI_Recv(&task, sizeof(WorkData), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    processTaskParallel(task, slaveEnoughStates);
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

    void processTaskParallel(WorkData task, const int slaveEnoughStates) {
        queue<WorkData> bfsQueue;
        bfsQueue.push(task);
        vector<WorkData> work;

        while (!bfsQueue.empty() && bfsQueue.size() + work.size() < slaveEnoughStates) {
            auto [node, p] = bfsQueue.front();
            bfsQueue.pop();
            generateNextStatesBFS(node, p, bfsQueue, work);
        }
        while (!bfsQueue.empty()) {
            work.push_back(bfsQueue.front());
            bfsQueue.pop();
        }

        const int workSize = work.size();

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < workSize; i++) {
            solveRecursive(work[i].p, work[i].node);
        }
    }

    void solveRecursive(Coordinates p, Node& current) {
        int currentBest;
        #pragma omp atomic read relaxed
        currentBest = bestPriceShared;

        thread_local int calls = 0;
        if ((++calls % 8192) == 0) {
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
                currentBest = new_min;
            }
        }

        if (current.price >= currentBest || currentBest == trivialBound) return;
        if (abs(current.quatrominoCntPerType[Z] - current.quatrominoCntPerType[T]) - 1 > current.quatrominoCntPerType[NOT_DECIDED] / 4) return;

        if (p.r >= R) {
            if (abs(current.quatrominoCntPerType[Z] - current.quatrominoCntPerType[T]) <= 1) {
                if (current.price < currentBest) {
                    #pragma omp critical(update_best)
                    {
                        if (current.price < bestPriceShared) {
                            #pragma omp atomic write
                            bestPriceShared = current.price;
                            bestSolution = current.solution;

                            MPI_Send(&bestPriceShared, 1, MPI_INT, 0, TAG_NEW_MIN, MPI_COMM_WORLD);
                        }
                    }
                }
            }
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                putShape(current, shape, p.r, p.c);
                solveRecursive(nextNotDecidedPosition(p, current.solution), current);
                removeShape(current, shape, p.r, p.c);
            }
        }

        if (tryPutClear(current, p)) {
            solveRecursive(nextNotDecidedPosition(p, current.solution), current);
            removeClear(current, p);
        }
    }

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

    void generateNextStatesBFS(Node current, Coordinates p, queue<WorkData>& q, vector<WorkData>& items) {
        if (p.r >= R) {
            items.push_back({current, p});
            return;
        }

        for (const auto& shape : ShapesUpperLeft) {
            if (canPutShape(current, shape, p.r, p.c, NOT_DECIDED)) {
                Node nextNode = current;
                putShape(nextNode, shape, p.r, p.c);
                q.push({nextNode, nextNotDecidedPosition(p, nextNode.solution)});
            }
        }

        Node nextStateClear = current;
        if (tryPutClear(nextStateClear, p)) {
            q.push({nextStateClear, nextNotDecidedPosition(p, nextStateClear.solution)});
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
};

int main(int argc, char* argv[]) {
        if (argc != 2) {
            cerr << "Usage: " << argv[0] << " <number-of-threads>" << endl;
            return 1;
        }
        const int numThreads = std::stoi(argv[1]);
        if (numThreads <= 0) {
            cerr << "Error: Number of threads must be greater than 0." << endl;
            return 1;
        }

        omp_set_num_threads(numThreads);
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

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        solver.solve(ENOUGH_STATES_COEF_MASTER * (size - 1), ENOUGH_STATES_COEF_SLAVE * numThreads);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if (rank == 0) {
            cout << "Cas behu: " << (end_time - start_time) << " sekund" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}