using System;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;

public class HybridAStarPlanner : MonoBehaviour
{
    [Header("Vehicle Reference")]
    public CarController carController;

    [Header("Vehicle Kinematics")]
    private float wheelBase;
    private float maxSteeringAngle;
    public float stepSize = 0.5f; 
    
    [Header("Grid Resolution (Closed List)")]
    public float xyResolution = 0.5f;
    public float yawResolution = 5f; // Degrees

    [Header("Environment & Collision")]
    public LayerMask obstacleLayer;

    [HideInInspector]
    public Vector3 vehicleSize;

    public class HybridNode : IComparable<HybridNode>
    {
        public Vector3 Position;
        public float Yaw; // Degrees
        public float GCost;
        public float FCost;
        public HybridNode Parent;
        public bool IsReverse;
        public float Steering;

        public int gridX, gridZ, gridYaw;

        public HybridNode(Vector3 pos, float yaw, float gCost, float fCost, HybridNode parent, bool isReverse, float steering, float xyRes, float yawRes)
        {
            Position = pos;
            Yaw = yaw;
            GCost = gCost;
            FCost = fCost;
            Parent = parent;
            IsReverse = isReverse;
            Steering = steering;

            // Discretize continuous state into a 3D grid index to prevent infinite loops
            gridX = Mathf.RoundToInt(pos.x / xyRes);
            gridZ = Mathf.RoundToInt(pos.z / xyRes);
            
            float normalizedYaw = NormalizeAngle(yaw);
            gridYaw = Mathf.RoundToInt(normalizedYaw / yawRes);
        }

        public string GetStateKey()
        {
            return $"{gridX}_{gridZ}_{gridYaw}";
        }

        public int CompareTo(HybridNode other)
        {
            return FCost.CompareTo(other.FCost);
        }
    }

    private static float NormalizeAngle(float angle)
    {
        angle %= 360f;
        if (angle < 0) angle += 360f;
        return angle;
    }

    private void CalculateDynamicVehicleSize()
    {
        if (carController == null) return;

        BoxCollider bodyCollider = carController.GetComponentInChildren<BoxCollider>();
        
        if (bodyCollider != null)
        {
            float exactWidth = Mathf.Abs(bodyCollider.size.x * bodyCollider.transform.localScale.x);
            float exactHeight = Mathf.Abs(bodyCollider.size.y * bodyCollider.transform.localScale.y);
            float exactLength = Mathf.Abs(bodyCollider.size.z * bodyCollider.transform.localScale.z);
            
            // Add 0.4m safety margin to force the path clear of parked car corners
            float margin = 0.4f;
            vehicleSize = new Vector3(exactWidth + margin, exactHeight, exactLength + margin);
        }
        else
        {
            vehicleSize = new Vector3(1.92f + 0.4f, 1.2f, 4.32f + 0.4f); 
        }
    }

    public List<Pose> GeneratePath(Vector3 startPos, float startYaw, Vector3 targetPos, float targetYaw)
    {
        MinHeap<HybridNode> openSet = new MinHeap<HybridNode>();
        HashSet<string> closedSet = new HashSet<string>();

        float initialHeuristic = CalculateHeuristic(startPos, startYaw, targetPos, targetYaw);
        HybridNode startNode = new HybridNode(startPos, startYaw, 0, initialHeuristic, null, false, 0, xyResolution, yawResolution);
        
        openSet.Add(startNode);

        int iterations = 0;
        int maxIterations = 50000;

        if (carController != null)
        {
            wheelBase = carController.WheelBase;
            maxSteeringAngle = carController.MaxSteeringAngle;
            if (wheelBase <= 0.1f) 
            {
                wheelBase = 2.5f;
                Debug.LogWarning("WheelBase was 0. Using fallback of 2.5f.");
            }
            
            CalculateDynamicVehicleSize();
        
        }
        else
        {
            Debug.LogError("CarController not assigned to HybridAStarPlanner!");
            return new List<Pose>();
        }

        float[] steeringAngles = { -maxSteeringAngle, 0, maxSteeringAngle };
        int[] directions = { 1, -1 }; // 1 = forward, -1 = reverse

        while (openSet.Count > 0 && iterations < maxIterations)
        {
            iterations++;
            HybridNode current = openSet.ExtractMin();
            string stateKey = current.GetStateKey();

            if (closedSet.Contains(stateKey)) continue;
            closedSet.Add(stateKey);

            // Goal Tolerance Check (Translational and Rotational)
            if (Vector3.Distance(current.Position, targetPos) < 2.5f && Mathf.Abs(Mathf.DeltaAngle(current.Yaw, targetYaw)) < 35f)
                {
                    return ReconstructPath(current);
                }

            // Kinematic Expansion
            foreach (int dir in directions)
            {
                foreach (float steer in steeringAngles)
                {
                    // Apply Bicycle Model kinematics
                    float distance = dir * stepSize;
                    float deltaYaw = (distance / wheelBase) * Mathf.Tan(steer * Mathf.Deg2Rad) * Mathf.Rad2Deg;
                    
                    float nextYaw = current.Yaw + deltaYaw;
                    // Note: Unity coordinates. X is right/left, Z is forward/back
                    Vector3 nextPos = current.Position + new Vector3(Mathf.Sin(nextYaw * Mathf.Deg2Rad), 0, Mathf.Cos(nextYaw * Mathf.Deg2Rad)) * distance;

                    if (IsCollision(nextPos, nextYaw)) continue;

                    // Cost Calculation
                    bool isReverse = dir < 0;
                    float moveCost = stepSize;
                    
                    if (isReverse) moveCost *= 1.5f; // Penalize driving backward
                    if (current.IsReverse != isReverse) moveCost *= 2.0f; // Penalize gear shifting
                    if (Mathf.Abs(steer) > 0) moveCost *= 1.2f; // Penalize steering to encourage straight lines

                    float gCost = current.GCost + moveCost;
                    float hCost = CalculateHeuristic(nextPos, nextYaw, targetPos, targetYaw);
                    
                    HybridNode nextNode = new HybridNode(nextPos, nextYaw, gCost, gCost + hCost, current, isReverse, steer, xyResolution, yawResolution);

                    if (!closedSet.Contains(nextNode.GetStateKey()))
                    {
                        openSet.Add(nextNode);
                    }
                }
            }
        }

        Debug.LogWarning($"Hybrid A* Failed: Reached {iterations} iterations without finding target.");
        return new List<Pose>();
    }

    private float CalculateHeuristic(Vector3 pos, float yaw, Vector3 targetPos, float targetYaw)
    {
        // Base Euclidean distance
        float distance = Vector3.Distance(pos, targetPos);
        float angleDiff = Mathf.Abs(Mathf.DeltaAngle(yaw, targetYaw));
        
        // Combine translational distance and heading alignment
        return distance + (angleDiff * 0.05f); 
    }

    private bool IsCollision(Vector3 pos, float yaw)
    {
        // If the root transform is at the ground or rear axle, offset to the true center
        Vector3 boxCenter = pos + Vector3.up * (vehicleSize.y / 2f);

        Collider[] hitColliders = Physics.OverlapBox(
            boxCenter, 
            vehicleSize / 2f, // Half-extents
            Quaternion.Euler(0, yaw, 0), 
            obstacleLayer,
            QueryTriggerInteraction.Ignore
        );

        return hitColliders.Length > 0;
    }

    private List<Pose> ReconstructPath(HybridNode node)
    {
        List<Pose> path = new List<Pose>();
        while (node != null)
        {
            // Store both position and vehicle orientation at this node
            path.Add(new Pose(node.Position, Quaternion.Euler(0f, node.Yaw, 0f)));
            node = node.Parent;
        }
        path.Reverse();
        return path;
    }
}

// Custom Priority Queue for fast O(log n) node extraction
public class MinHeap<T> where T : IComparable<T>
{
    private List<T> elements = new List<T>();

    public int Count => elements.Count;

    public void Add(T item)
    {
        elements.Add(item);
        HeapifyUp(elements.Count - 1);
    }

    public T ExtractMin()
    {
        if (elements.Count == 0) throw new InvalidOperationException("Heap is empty");
        
        T min = elements[0];
        elements[0] = elements[elements.Count - 1];
        elements.RemoveAt(elements.Count - 1);
        
        if (elements.Count > 0)
        {
            HeapifyDown(0);
        }
        
        return min;
    }

    private void HeapifyUp(int index)
    {
        int parentIndex = (index - 1) / 2;
        while (index > 0 && elements[index].CompareTo(elements[parentIndex]) < 0)
        {
            Swap(index, parentIndex);
            index = parentIndex;
            parentIndex = (index - 1) / 2;
        }
    }

    private void HeapifyDown(int index)
    {
        int smallest = index;
        while (true)
        {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;

            if (leftChild < elements.Count && elements[leftChild].CompareTo(elements[smallest]) < 0)
                smallest = leftChild;

            if (rightChild < elements.Count && elements[rightChild].CompareTo(elements[smallest]) < 0)
                smallest = rightChild;

            if (smallest == index) break;

            Swap(index, smallest);
            index = smallest;
        }
    }

    private void Swap(int a, int b)
    {
        T temp = elements[a];
        elements[a] = elements[b];
        elements[b] = temp;
    }
}