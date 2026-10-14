using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace UnityStandardAssets.Vehicles.Car{
    public class CarAgent : Agent
    {
        public float spawnRadiusX = 2f;
        public float spawnRadiusZ = 2f;
        public float envRadiusX = 3f;
        public float envRadiusZ = 10f;
        public float inTargetMultiplier = 1.5f;
        public GameObject target;
        public Camera[] carCameras;
        public int cameraWidth = 200;
        public int cameraHeight = 200;
        public bool cameraGrayScale = false;
        public SensorCompressionType sensorCompressionType = SensorCompressionType.PNG;

        private CarController carController;
        EnvironmentParameters defaultParameters;
        private Rigidbody rb;
        private int steps = 0;

        private bool inTarget = false;

        private Vector3 startPosition;
        private Quaternion startRotation;
        private Vector3 lastPosition;

        // Automated parking detection variables
        public bool findParkingSpot = true;
        private bool isLookingForSpot;
        private bool isPositioning;
        private RayPerceptionSensorComponent3D RayPerceptionSensorComponent;
        private Vector3 detectedSpotLocation;
        private float predictedSpotSize = 0f;

        // Hybrid A* Integration variables
        [Header("Hybrid A* Integration")]
        public HybridAStarPlanner pathPlanner;
        private List<Pose> plannedPath = new List<Pose>();
        private bool isPathGenerated = false;
        private int currentWaypointIndex = 0;

        void FixedUpdate(){
            // If Looking for spot is enabled, the car will drive and try to find a spot first
            if(isLookingForSpot){
                CruiseControl(4f);
                FindParkingSpot();
            }

            // PositionCar has been removed, as SAC now handles positioning using the Hybrid A* baseline

            // Get desicion from python by requesting the next action
            if(!isLookingForSpot){
                RequestDecision();
                // If agent get's too par from it's target, stop the episode and reset the agent
                if(Mathf.Abs(transform.position.x - target.transform.position.x) > envRadiusX || Mathf.Abs(transform.position.z - target.transform.position.z) > envRadiusZ){
                    AddReward(-100f);
                    EndEpisode();
                }
            }  
        }

        private void Reset(){
            if(findParkingSpot){
                isLookingForSpot = true;
                isPositioning = false;
            } else {
                isLookingForSpot = false;
                isPositioning = true;
            }
            
            // Clear previous trajectory for the new episode
            plannedPath.Clear();
            isPathGenerated = false;
            currentWaypointIndex = 0;

            // Spawn randomly in defined range
            float spawnX = Random.Range(startPosition.x - spawnRadiusX, startPosition.x + spawnRadiusX);
            float spawnZ = Random.Range(startPosition.z - spawnRadiusZ, startPosition.z + spawnRadiusZ);
            Vector3 spawnPosition = new Vector3(spawnX, startPosition.y, spawnZ);

            rb.transform.position = spawnPosition;
            rb.transform.rotation = startRotation;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            steps = 0;

            // Generate the path immediately from the spawn position if we aren't scanning
            if (!isLookingForSpot && pathPlanner != null && target != null) {
                float startYaw = rb.transform.eulerAngles.y;
                float targetYaw = target.transform.eulerAngles.y;
                
                // Calculate the path from the new random spawn position to the target
                plannedPath = pathPlanner.GeneratePath(transform.position, startYaw, target.transform.position, targetYaw);
                isPathGenerated = (plannedPath != null && plannedPath.Count > 0);
                currentWaypointIndex = 0;
            }
        }

        public override void Initialize(){
            carController = GetComponent<CarController>();
            rb = GetComponent<Rigidbody>();

            isLookingForSpot = findParkingSpot;

            RayPerceptionSensorComponent = GetComponent<RayPerceptionSensorComponent3D>();

            defaultParameters = Academy.Instance.EnvironmentParameters;
            
            startPosition = transform.position;
            startRotation = transform.rotation;
            
            lastPosition = startPosition;

            Reset();
            AddCameras();
        }

        private void FindParkingSpot(){
            var RpMeasurements = RayPerceptionMeasurements();
            
            int LeftLikelihoodScore = 0;
            int RightLikelihoodScore = 0;

            // First check if the left and right perpendicular sensors are detecting a long range
            if(RpMeasurements.RDistL[2] > 0.9f){
                LeftLikelihoodScore += 1;
            }
            if(RpMeasurements.RDistR[2] > 0.9f){
                RightLikelihoodScore += 1;
            }

            // Sum the distances of the sensors for each side to check which side is just seeing space
            if(RpMeasurements.RDistL.Sum() < RpMeasurements.RDistR.Sum()){
                LeftLikelihoodScore += 1;
            }
            else{
                RightLikelihoodScore += 1;
            }

            // Check if sensor observations are symmetrical. This indicates the agent is in the middle of the parking space
            float RayDiff1 = Mathf.Abs(RpMeasurements.RDistL[0] - RpMeasurements.RDistL[4]);
            float RayDiff2 = Mathf.Abs(RpMeasurements.RDistL[1] - RpMeasurements.RDistL[3]);
            float TotalRayDiff = RayDiff1 + RayDiff2;

            if(TotalRayDiff < 0.1f){
                LeftLikelihoodScore += 1;
            }

            RayDiff1 = Mathf.Abs(RpMeasurements.RDistR[0] - RpMeasurements.RDistR[4]);
            RayDiff2 = Mathf.Abs(RpMeasurements.RDistR[1] - RpMeasurements.RDistR[3]);
            TotalRayDiff = RayDiff1 + RayDiff2;

            if(TotalRayDiff < 0.1f){
                RightLikelihoodScore += 1;
            }
            
            // Can be refactored
            // If one of the sides met all the requirements, the agent is in the middle of a space
            if(LeftLikelihoodScore == 3){
                // Validate the spot is lage enough
                float PredictedSpace = (RpMeasurements.RDistL[1] * Mathf.Cos(60*Mathf.Deg2Rad)) + (RpMeasurements.RDistL[3] * Mathf.Cos(60*Mathf.Deg2Rad));
                // Distances are in normalised units, so multiply by the ray length to get the actual distance
                PredictedSpace *= 7;


                if(PredictedSpace > 3f){
                    isLookingForSpot = false;
                    isPositioning = true;
                    predictedSpotSize = PredictedSpace;
                    detectedSpotLocation = new Vector3(transform.position.x, transform.position.y, transform.position.z);
                    Debug.Log("Found spot left");

                    // Generate Hybrid A* Path
                    if (pathPlanner != null) {
                        float startYaw = transform.eulerAngles.y;
                        float targetYaw = target.transform.eulerAngles.y;
                        plannedPath = pathPlanner.GeneratePath(transform.position, startYaw, target.transform.position, targetYaw);
                        isPathGenerated = (plannedPath != null && plannedPath.Count > 0);
                        currentWaypointIndex = 0;
                    }
                }
            }
            else if(RightLikelihoodScore == 3){
                // Validate the spot is lage enough
                float PredictedSpace = (RpMeasurements.RDistR[1] * Mathf.Cos(60*Mathf.Deg2Rad)) + (RpMeasurements.RDistR[3] * Mathf.Cos(60*Mathf.Deg2Rad));
                // Distances are in normalised units, so multiply by the ray length to get the actual distance
                PredictedSpace *= 7;
                
                if(PredictedSpace > 3f){
                    isLookingForSpot = false;
                    isPositioning = true;
                    predictedSpotSize = PredictedSpace;
                    detectedSpotLocation = new Vector3(transform.position.x, transform.position.y, transform.position.z);
                    Debug.Log("Found spot right");

                    // Generate Hybrid A* Path
                    if (pathPlanner != null) {
                        float startYaw = transform.eulerAngles.y;
                        float targetYaw = target.transform.eulerAngles.y;
                        plannedPath = pathPlanner.GeneratePath(transform.position, startYaw, target.transform.position, targetYaw);
                        isPathGenerated = (plannedPath != null && plannedPath.Count > 0);
                        currentWaypointIndex = 0;
                    }
                }
            }

            else{
                isLookingForSpot = true;
            }

        }

        private (float[] RDistL, float[] RDistR, float RDistF, float RDistB) RayPerceptionMeasurements(bool LogValues = false){
            RayPerceptionInput RayPerceptionIn = RayPerceptionSensorComponent.GetRayPerceptionInput();
            RayPerceptionOutput RayPerceptionOut = RayPerceptionSensor.Perceive(RayPerceptionIn, false);
            RayPerceptionOutput.RayOutput[] RayOutputs = RayPerceptionOut.RayOutputs;

            // ArRay length -1 because 2 Rays overlap in the end of the array when using 180 degree setting. Using 1 of the 2 is perfectly fine.
            int RayAmount = RayOutputs.Length - 1;
            float[] RayDistances = new float[RayAmount - 1];

            // Indexing is quite retarted.
            // Index ordering: [0] = front, [1] = right, [2] = left, [3] = right, [4] = left,...
            // Even numbers are left, odd numbers are right
            float[] RayDistancesLeft = new float[(RayAmount - 2) / 2];
            float[] RayDistancesRight = new float[(RayAmount - 2) / 2];

            float RayDistanceFront = RayOutputs[0].HitFraction;
            float RayDistanceBack = RayOutputs[RayAmount - 1].HitFraction;

            for(int i = 1; i < RayAmount-1; i++)
            {
                // If Even
                if(i % 2 == 0){
                    RayDistancesLeft[(i/2)-1] = RayOutputs[i].HitFraction;
                }
                // If Oneven
                else{
                    RayDistancesRight[(i-1)/2] = RayOutputs[i].HitFraction;
                }
            }

            if(LogValues){
                Debug.Log("Left rays:\n" + ArrayToString(RayDistancesLeft));
                Debug.Log("Right rays:\n" + ArrayToString(RayDistancesRight));
            }

            return (RayDistancesLeft, RayDistancesRight, RayDistanceFront, RayDistanceBack);
        }

        private void CruiseControl(float Speed){
            if(carController.CurrentSpeed < Speed){
                carController.Move(0, 0.5f, 0f, 0f);
            }
            else if(carController.CurrentSpeed > Speed){
                carController.Move(0, -0.5f, 0f, 0f);
            }
        }

        // Helper functions for debugging
        private string ArrayToString(float[] array){
            string str = "[";

            for(int i = 0; i < array.Length; i++){
                str += array[i] + "  ";
            }

            str += "]";

            return str;
        }

        // Same function but method overloaded for different types
        private string ArrayToString(int[] array){
            string str = "[";

            for(int i = 0; i < array.Length; i++){
                str += array[i] + "  ";
            }

            str += "]";

            return str;
        }

        private float CalculateReward(){
            
            // Compare the difference of the previous distance to target to the current one
            // If the agent got closer, reward it. Else penalize it.
            float reward = 0f;

            float totDirectionChangeReward = 0f;
            float totAngleChangeReward = 0f;
            float totDistanceReward = 0f;

            if(lastPosition != Vector3.zero){
                float distanceToTargetX = Mathf.Abs(transform.position.x - target.transform.position.x);
                float distanceToTargetZ = Mathf.Abs(transform.position.z - target.transform.position.z);

                float lastDistanceToTargetX = Mathf.Abs(lastPosition.x - target.transform.position.x);
                float lastDistanceToTargetZ = Mathf.Abs(lastPosition.z - target.transform.position.z);

                float directionChangeX = lastDistanceToTargetX - distanceToTargetX;
                float directionChangeZ = lastDistanceToTargetZ - distanceToTargetZ;

                totDirectionChangeReward = (directionChangeX + directionChangeZ) * 10f;
                totDirectionChangeReward = Mathf.Clamp(totDirectionChangeReward, -0.5f, 0.5f);

                float distanceRewardX = (1f - distanceToTargetX/envRadiusX);
                float distanceRewardZ = (1f - distanceToTargetZ/envRadiusZ);

                totDistanceReward = (distanceRewardX + distanceRewardZ) / 20f;

                reward += totDirectionChangeReward + totDistanceReward;
            }

            if(inTarget){
                float angleToTarget = Vector3.Angle(transform.forward, target.transform.forward);
                // When driving in the spot backwards, the angle to target is 180 degrees
                if(angleToTarget > 90f){
                    angleToTarget = 180f - angleToTarget;
                }

                angleToTarget = Mathf.Clamp(angleToTarget, 0f, 90f);
                float angleReward = (-(1f/45f) * angleToTarget) + 1f;

                totAngleChangeReward = angleReward + 1f;

                // Reward for minimising the angle to the target
                reward += totAngleChangeReward;

                float distanceToTarget = Vector3.Distance(transform.position, target.transform.position);

                // Check if car was able to park and reward it accordingly
                if(angleToTarget < 2.5f && distanceToTarget < 1f && Mathf.Abs(carController.CurrentSpeed) < 2f){
                    Debug.Log("Car parked!");
                    reward += 100f;
                    EndEpisode();
                }
            }

            lastPosition = transform.position;
            return reward;            
        }

        private void AddCameras(){
            // Add car's cameras to the agent
            foreach(Camera c in carCameras){
                // First fetch the camera's gameobject and it's name
                GameObject cameraGameObject = c.gameObject;
                string cameraName = cameraGameObject.name;
            
                // Add cameraSensor to gameObject
                CameraSensorComponent cameraSensorComponent = this.gameObject.AddComponent<CameraSensorComponent>();
                cameraSensorComponent.Camera = c;
                cameraSensorComponent.SensorName = cameraName;
                cameraSensorComponent.Width = cameraWidth;
                cameraSensorComponent.Height = cameraHeight;
                cameraSensorComponent.Grayscale = cameraGrayScale;
                cameraSensorComponent.ObservationStacks = 1;
                cameraSensorComponent.ObservationType = ObservationType.Default;
                cameraSensorComponent.CompressionType = sensorCompressionType;
            }
        }

        public override void OnEpisodeBegin(){
            Reset();
        }

        public override void CollectObservations(VectorSensor sensor){
            sensor.AddObservation(carController.CurrentSpeed);

            // Add Hybrid A* tracking states for SAC to learn corrections
            if (isPathGenerated && currentWaypointIndex < plannedPath.Count) {
                Vector3 targetWaypoint = plannedPath[currentWaypointIndex].position;
                Vector3 localTarget = transform.InverseTransformPoint(targetWaypoint);

                sensor.AddObservation(localTarget.x); // crossTrackError
                sensor.AddObservation(localTarget.z); // distanceToWaypoint
            } else {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        private void UpdateWaypointProgression()
        {
            if (!isPathGenerated || plannedPath == null || plannedPath.Count == 0) return;
            
            // Don't advance past the final waypoint
            if (currentWaypointIndex >= plannedPath.Count - 1) return;

            float distanceToWaypoint = Vector3.Distance(transform.position, plannedPath[currentWaypointIndex].position);
            
            // If within 1.5 meters of the waypoint, target the next one
            if (distanceToWaypoint < 1.5f)
            {
                currentWaypointIndex++;
            }
        }

        public override void OnActionReceived(ActionBuffers actions){
            UpdateWaypointProgression();
            
            float steering = actions.ContinuousActions[0];
            float accel = actions.ContinuousActions[1];
            float reverse = actions.ContinuousActions[2];

            // Input is from -1 to 1, map values accordingly
            accel = (accel + 1) / 2;
            reverse = (reverse + 1) / 2;

            accel = accel - reverse;
            
            if(!isLookingForSpot){
                carController.Move(steering, accel, 0f, 0f);
            }
            
            steps++;

            float reward = CalculateReward();
            AddReward(reward);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            ActionSegment<float> continuousActionsOut = actionsOut.ContinuousActions;

            float steering = Input.GetAxis("Horizontal"); //-1 to 1
            float accel = Input.GetAxis("Accelerate");  //0 to 1
            float reverse = Input.GetAxis("Reverse");   //0 to 1

            // Input from network is between -1 to 1, map values accordingly
            accel = accel * 2 - 1;
            reverse = reverse * 2 - 1;

            continuousActionsOut[0] = steering;
            continuousActionsOut[1] = accel;
            continuousActionsOut[2] = reverse;
        }

        void OnTriggerEnter(Collider other)
        {
            if(other.gameObject.tag == "Finish"){
                inTarget = true;
            }
        }

        void OnTriggerExit(Collider other)
        {
            if(other.gameObject.tag == "Finish"){
                inTarget = false;
            }
        }

        void OnCollisionEnter(Collision collision)
        {
            print(collision.gameObject.tag);
            if (collision.gameObject.tag == "Wall")
            {
                AddReward(-10f);
                EndEpisode();
            }
        }

        void OnCollisionStay(Collision collision)
        {
            if (collision.gameObject.tag == "Kerb")
            {
                AddReward(-2f);
            }
            else if(collision.gameObject.tag == "Car")
            {
                
                float reward = -Mathf.Abs(carController.CurrentSpeed) * 50f - 5f;
                AddReward(reward);
                EndEpisode();
            }
        }

        // Visualize the Hybrid A* path in the Unity Editor
        private void OnDrawGizmos()
        {
            if (!isPathGenerated || plannedPath == null || plannedPath.Count == 0) return;

            // Subtract 0.4f safety margin to match true vehicle size
            Vector3 boxDimensions = (pathPlanner != null && pathPlanner.vehicleSize != Vector3.zero) 
                ? new Vector3(pathPlanner.vehicleSize.x - 0.4f, pathPlanner.vehicleSize.y, pathPlanner.vehicleSize.z - 0.4f)
                : new Vector3(1.92f, 1.2f, 4.32f);

            // Save initial Gizmos matrix
            Matrix4x4 defaultMatrix = Gizmos.matrix;

            for (int i = 0; i < plannedPath.Count; i++)
            {
                Pose waypointPose = plannedPath[i];

                // 1. Draw waypoint sphere and trajectory line
                Gizmos.matrix = defaultMatrix;
                Gizmos.color = Color.cyan;
                Gizmos.DrawSphere(waypointPose.position, 0.15f);

                if (i < plannedPath.Count - 1)
                {
                    Gizmos.DrawLine(waypointPose.position, plannedPath[i + 1].position);
                }

                // 2. Draw vehicle bounding footprint at each waypoint
                // Sample every 2nd or 3rd node if visualization gets too crowded
                Gizmos.matrix = Matrix4x4.TRS(
                    waypointPose.position + Vector3.up * (boxDimensions.y / 2f), 
                    waypointPose.rotation, 
                    Vector3.one
                );

                // Highlight the current targeted waypoint in yellow, others in transparent green
                if (i == currentWaypointIndex)
                {
                    Gizmos.color = Color.yellow;
                }
                else
                {
                    Gizmos.color = new Color(0f, 1f, 0.4f, 0.35f);
                }

                Gizmos.DrawWireCube(Vector3.zero, boxDimensions);
            }

            // Restore matrix
            Gizmos.matrix = defaultMatrix;
        }
    }
}