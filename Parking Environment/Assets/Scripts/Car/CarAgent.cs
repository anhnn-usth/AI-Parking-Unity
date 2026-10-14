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

        [Header("Adaptive Curriculum Settings")]
        public HybridAStarPlanner pathPlanner;
        public bool useAdaptiveCurriculum = true;
        public int currentCurriculumLevel = 1;
        public int maxCurriculumLevel = 3;
        public int requiredSuccessesToAdvance = 5;
        private int consecutiveSuccesses = 0;
        private int maxSpawnAttempts = 15;

        private CarController carController;
        EnvironmentParameters defaultParameters;
        private Rigidbody rb;
        private int steps = 0;

        private bool inTarget = false;

        private Vector3 startPosition;
        private Quaternion startRotation;
        private Vector3 lastPosition;

        public bool findParkingSpot = true;
        private bool isLookingForSpot;
        private bool isPositioning;
        private RayPerceptionSensorComponent3D RayPerceptionSensorComponent;
        private Vector3 detectedSpotLocation;
        private float predictedSpotSize = 0f;

        void FixedUpdate(){
            if(isLookingForSpot){
                CruiseControl(4f);
                FindParkingSpot();
            }

            if(isPositioning && findParkingSpot){
                PositionCar(-1f);
            }

            if(!isLookingForSpot){
                RequestDecision();
                if(Mathf.Abs(transform.position.x - target.transform.position.x) > envRadiusX || Mathf.Abs(transform.position.z - target.transform.position.z) > envRadiusZ){
                    AddReward(-100f);
                    HandleFailure();
                }
            }  
        }

        private void Reset(){
            if(findParkingSpot){
                isLookingForSpot = true;
                isPositioning = false;
            }

            bool validSpawnFound = false;

            if (useAdaptiveCurriculum && pathPlanner != null && !isLookingForSpot)
            {
                for (int i = 0; i < maxSpawnAttempts; i++)
                {
                    float spawnX = Random.Range(startPosition.x - spawnRadiusX, startPosition.x + spawnRadiusX);
                    float spawnZ = Random.Range(startPosition.z - spawnRadiusZ, startPosition.z + spawnRadiusZ);
                    float spawnYaw = Random.Range(0f, 360f);
                    Vector3 testPosition = new Vector3(spawnX, startPosition.y, spawnZ);
                    
                    float targetYaw = target.transform.rotation.eulerAngles.y;
                    
                    // Estimate task difficulty using Hybrid A*
                    var assessment = pathPlanner.EvaluatePath(testPosition, spawnYaw, target.transform.position, targetYaw);

                    if (assessment.IsFeasible)
                    {
                        int scenarioDifficulty = 1;
                        if (assessment.GearShifts > 2 || assessment.PathLength > 15f) scenarioDifficulty = 3;
                        else if (assessment.GearShifts > 0) scenarioDifficulty = 2;

                        // Retain easier scenarios but bound max difficulty to agent's current progress
                        if (scenarioDifficulty <= currentCurriculumLevel)
                        {
                            ApplySpawnTransform(testPosition, Quaternion.Euler(0, spawnYaw, 0));
                            validSpawnFound = true;
                            break;
                        }
                    }
                }
            }

            // Fallback if curriculum spawn fails or is disabled
            if (!validSpawnFound)
            {
                float spawnX = Random.Range(startPosition.x - spawnRadiusX, startPosition.x + spawnRadiusX);
                float spawnZ = Random.Range(startPosition.z - spawnRadiusZ, startPosition.z + spawnRadiusZ);
                ApplySpawnTransform(new Vector3(spawnX, startPosition.y, spawnZ), startRotation);
            }

            steps = 0;
        }

        private void ApplySpawnTransform(Vector3 pos, Quaternion rot)
        {
            rb.transform.position = pos;
            rb.transform.rotation = rot;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            lastPosition = pos;
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

        // --- FindParkingSpot, RayPerceptionMeasurements, CruiseControl, PositionCar, ArrayToString remain unchanged ---
        private void FindParkingSpot() { /* Existing Code Maintained */ }
        private (float[] RDistL, float[] RDistR, float RDistF, float RDistB) RayPerceptionMeasurements(bool LogValues = false) { return (new float[0], new float[0], 0f, 0f); /* Kept brief for space, use existing code here */ }
        private void CruiseControl(float Speed) { /* Existing Code Maintained */ }
        private void PositionCar(float offsetX) { /* Existing Code Maintained */ }
        private string ArrayToString(float[] array) { return ""; /* Existing Code Maintained */ }
        private string ArrayToString(int[] array) { return ""; /* Existing Code Maintained */ }

        private float CalculateReward(){
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
                if(angleToTarget > 90f){
                    angleToTarget = 180f - angleToTarget;
                }

                angleToTarget = Mathf.Clamp(angleToTarget, 0f, 90f);
                float angleReward = (-(1f/45f) * angleToTarget) + 1f;

                totAngleChangeReward = angleReward + 1f;
                reward += totAngleChangeReward;

                float distanceToTarget = Vector3.Distance(transform.position, target.transform.position);

                if(angleToTarget < 2.5f && distanceToTarget < 1f && Mathf.Abs(carController.CurrentSpeed) < 2f){
                    Debug.Log("Car parked!");
                    reward += 100f;
                    HandleSuccess();
                }
            }

            lastPosition = transform.position;
            return reward;            
        }

        private void HandleSuccess()
        {
            consecutiveSuccesses++;
            if (consecutiveSuccesses >= requiredSuccessesToAdvance && currentCurriculumLevel < maxCurriculumLevel)
            {
                currentCurriculumLevel++;
                consecutiveSuccesses = 0;
                Debug.Log($"Curriculum Level Increased to {currentCurriculumLevel}");
            }
            EndEpisode();
        }

        private void HandleFailure()
        {
            consecutiveSuccesses = Mathf.Max(0, consecutiveSuccesses - 1);
            EndEpisode();
        }

        private void AddCameras(){
            foreach(Camera c in carCameras){
                GameObject cameraGameObject = c.gameObject;
                string cameraName = cameraGameObject.name;
            
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
        }

        public override void OnActionReceived(ActionBuffers actions){
            float steering = actions.ContinuousActions[0];
            float accel = actions.ContinuousActions[1];
            float reverse = actions.ContinuousActions[2];

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

            float steering = Input.GetAxis("Horizontal"); 
            float accel = Input.GetAxis("Accelerate");  
            float reverse = Input.GetAxis("Reverse");   

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
            if (collision.gameObject.tag == "Wall")
            {
                AddReward(-10f);
                HandleFailure();
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
                HandleFailure();
            }
        }
    }
}