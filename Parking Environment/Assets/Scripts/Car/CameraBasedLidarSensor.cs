using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.MLAgents;

namespace UnityStandardAssets.Vehicles.Car
{
    /// <summary>
    /// Camera-Based 3D LiDAR sensor for the ego car.
    /// Emulates a multi-channel 3D LiDAR using standard Unity Cameras and 3D radial depth rendering,
    /// outputting 3D Point Cloud (x, y, z) coordinates for every laser beam in the ego car's local 3D frame.
    /// </summary>
    public class CameraBasedLidarSensor : MonoBehaviour
    {
        public enum LidarSamplingMode
        {
            /// <summary>
            /// Samples the 3D scene using Unity Cameras and the Hidden/CameraBasedLidarDepth replacement shader
            /// (automatically falls back to PhysicsRaycast when running in headless -nographics mode).
            /// </summary>
            CameraDepthShader = 0,

            /// <summary>
            /// Samples the 3D scene using 3D physics raycasts along each camera's frustum rays.
            /// </summary>
            PhysicsRaycast = 1
        }

        [Header("3D LiDAR Camera Rig")]
        [Tooltip("Cameras forming the 360° 3D LiDAR rig (defaults to the 8 surround cameras under Self_driving_cameras).")]
        public Camera[] lidarCameras;

        [Header("3D Scan Pattern & Geometry")]
        [Tooltip("Number of horizontal laser samples per camera (e.g., 16 samples x 8 cameras = 128 horizontal azimuth samples).")]
        [Range(1, 256)]
        public int horizontalSamplesPerCamera = 16;

        [Tooltip("Number of vertical laser channels (elevation layers) per camera for 3D scanning.")]
        [Range(2, 64)]
        public int verticalChannels = 4;

        [Tooltip("Horizontal field of view in degrees per camera.")]
        [Range(10f, 160f)]
        public float horizontalFov = 90f;

        [Tooltip("Minimum vertical elevation angle in degrees.")]
        [Range(-60f, 0f)]
        public float minVerticalAngle = -10f;

        [Tooltip("Maximum vertical elevation angle in degrees.")]
        [Range(0f, 60f)]
        public float maxVerticalAngle = 15f;

        [Header("Ground & Trigger Filtering")]
        [Tooltip("Exclude flat ground Plane and non-solid trigger objects (such as the green Target parking zone) from 3D LiDAR obstacle returns.")]
        public bool ignoreGroundAndTriggers = true;

        [Tooltip("Height threshold (in meters relative to ego car base) below which hits are filtered as road surface.")]
        [Range(-0.2f, 0.5f)]
        public float groundHeightThreshold = 0.15f;

        [Header("Range & Timing")]
        [Tooltip("Minimum 3D detection range in meters (near clip plane).")]
        public float minRange = 0.3f;

        [Tooltip("Maximum 3D detection range in meters (far clip plane).")]
        public float maxRange = 20f;

        [Tooltip("3D LiDAR scan update rate in Hz (revolutions/scans per second). Standard automotive 3D LiDAR is 10 Hz.")]
        public float rotationFrequencyHz = 10f;

        [Header("Sampling & Noise Configuration")]
        [Tooltip("3D sampling implementation: GPU Camera Depth Shader (Raster 3D LiDAR) or 3D Physics Raycast.")]
        public LidarSamplingMode samplingMode = LidarSamplingMode.CameraDepthShader;

        [Tooltip("Pool vertical elevation channels per azimuth column into the closest 3D (x, y, z) obstacle point so native SB3 MlpPolicy receives a compact 3D point cloud (128x3 = 384 floats instead of 512x3 = 1536).")]
        public bool poolVerticalChannelsForMlp = true;

        [Tooltip("Standard deviation of Gaussian radial range noise in meters (0 = noise-free).")]
        [Range(0f, 0.5f)]
        public float rangeNoiseStdDev = 0f;

        [Tooltip("Layer mask of 3D objects detectable by the LiDAR.")]
        public LayerMask cullingMask = ~0;

        [Header("3D Point Cloud Visualization")]
        [Tooltip("Automatically bypass 3D point-cloud GUI and mesh rendering when connected to a Python ML-Agents trainer to maximize training speed and guarantee zero visual overhead.")]
        public bool autoDisableVisualsDuringTraining = true;

        [Tooltip("Render 3D point cloud spheres directly in the 3D scene (visible only to Main Camera, never to LiDAR cameras).")]
        public bool render3DPointCloud = true;

        [Tooltip("Show the interactive 3D Perspective Point-Cloud Viewer HUD in the Game View (drag with mouse to orbit in 3D).")]
        public bool show3DPointCloudViewer = true;

        [Tooltip("Draw vertical elevation stalks under each 3D point in the 3D Viewer to highlight Y-height across vertical channels.")]
        public bool show3DElevationStalks = true;

        [Tooltip("Draw 3D laser beam lines from the cameras to the 3D hit points.")]
        public bool drawDebugRays = false;

        [Tooltip("Radius of each rendered 3D point cloud sphere in meters.")]
        [Range(0.02f, 0.3f)]
        public float pointCloudSphereRadius = 0.1f;

        private Shader lidarDepthShader;
        private RenderTexture scanRenderTexture;
        private Texture2D readbackTexture;
        private bool useRFloatTexture;

        private float[] normalizedRanges;
        private float[] observationBuffer;
        private Vector3[] localPointCloudXYZ;
        private Vector3[] worldPointCloudXYZ;
        private Vector3[] lastWorldRayOrigins;
        private Vector3[] lastWorldRayDirections;
        private Vector3[] cachedLocalRayOrigins;
        private Vector3[] cachedLocalRayDirections;
        private float[] lastHitDistances;

        private Collider[] egoColliders;
        private bool[] egoColliderStates;

        private Renderer[] egoRenderers;
        private bool[] egoRendererStates;
        private readonly List<Renderer> ignoredSceneRenderers = new List<Renderer>();
        private readonly List<bool> ignoredSceneRendererStates = new List<bool>();

        private float lastScanTime = -999f;
        private bool isInitialized = false;

        private static readonly int LidarMaxRangeShaderId = Shader.PropertyToID("_LidarMaxRange");
        private static readonly int BaseColorShaderId = Shader.PropertyToID("_BaseColor");
        private static readonly int ColorShaderId = Shader.PropertyToID("_Color");

        private static CameraBasedLidarSensor primaryVisualSensor;
        private static Mesh sharedSphereMesh;
        private static Material sharedPointMaterial;
        private MaterialPropertyBlock pointPropertyBlock;

        /// <summary>
        /// Number of active cameras in the 3D LiDAR rig.
        /// </summary>
        public int CameraCount
        {
            get
            {
                if (lidarCameras != null && lidarCameras.Length > 0)
                {
                    return lidarCameras.Length;
                }
                Camera[] found = FindRigCameras();
                return found != null && found.Length > 0 ? found.Length : 8;
            }
        }

        /// <summary>
        /// Total number of horizontal azimuth columns across all cameras (CameraCount * horizontalSamplesPerCamera).
        /// </summary>
        public int HorizontalAzimuthCount => Mathf.Max(1, CameraCount) * Mathf.Max(1, horizontalSamplesPerCamera);

        /// <summary>
        /// Total number of 3D laser beams across all cameras (CameraCount * horizontalSamplesPerCamera * verticalChannels).
        /// </summary>
        public int TotalBeamCount => HorizontalAzimuthCount * Mathf.Max(2, verticalChannels);

        /// <summary>
        /// Number of 3D (x, y, z) points written to the ML-Agents VectorSensor per step.
        /// When poolVerticalChannelsForMlp is true, each horizontal azimuth column outputs its closest 3D hit point (x, y, z)
        /// across all vertical elevation channels (128 points = 384 floats), ideal for native SB3 MlpPolicy.
        /// </summary>
        public int ObservationPointCount => poolVerticalChannelsForMlp ? HorizontalAzimuthCount : TotalBeamCount;

        /// <summary>
        /// Total number of floats written to the ML-Agents VectorSensor per step:
        /// 3 floats (local X, Y, Z) per exported 3D point (ObservationPointCount * 3).
        /// </summary>
        public int ObservationSize => ObservationPointCount * 3;

        /// <summary>
        /// Read-only access to the latest normalized radial range values [0, 1] for each 3D beam.
        /// </summary>
        public float[] LatestNormalizedRanges => normalizedRanges;

        /// <summary>
        /// Read-only access to the latest 3D Point Cloud coordinates (x, y, z) in ego-car local space (in meters).
        /// </summary>
        public Vector3[] LatestLocalPointCloudXYZ => localPointCloudXYZ;

        /// <summary>
        /// Read-only access to the latest 3D Point Cloud coordinates (x, y, z) in Unity world space.
        /// </summary>
        public Vector3[] LatestWorldPointCloudXYZ => worldPointCloudXYZ;

        /// <summary>
        /// Initializes the 3D LiDAR camera rig, render targets, and 3D point cloud buffers.
        /// </summary>
        public void Initialize(Camera[] overrideCameras = null)
        {
            if (overrideCameras != null && overrideCameras.Length > 0)
            {
                lidarCameras = overrideCameras;
            }
            else if (lidarCameras == null || lidarCameras.Length == 0)
            {
                lidarCameras = FindRigCameras();
            }

            ActivateCameraHierarchy();
            NormalizeNegativeScaleContainers();
            ConfigureCameras();

            // Cache ego car colliders and renderers to prevent self-hit
            egoColliders = GetComponentsInChildren<Collider>(true);
            egoColliderStates = new bool[egoColliders.Length];
            egoRenderers = GetComponentsInChildren<Renderer>(true);
            egoRendererStates = new bool[egoRenderers.Length];
            CacheIgnoredSceneRenderers();

            // Load 3D radial depth replacement shader
            lidarDepthShader = Shader.Find("Hidden/CameraBasedLidarDepth");
            if (lidarDepthShader == null)
            {
                lidarDepthShader = Resources.Load<Shader>("Shaders/CameraBasedLidarDepth");
            }

            AllocateBuffers();
            lastScanTime = -999f;
            isInitialized = true;
        }

        /// <summary>
        /// Resets the LiDAR scan timer at the start of an episode.
        /// Staggers the initial scan offset slightly across multiple agents to avoid frame spikes.
        /// </summary>
        public void ResetSensor()
        {
            if (rotationFrequencyHz > 0f)
            {
                lastScanTime = Time.time - (1f / rotationFrequencyHz) + UnityEngine.Random.Range(0f, 0.02f);
            }
            else
            {
                lastScanTime = -999f;
            }
        }

        /// <summary>
        /// Performs or retrieves the current 3D LiDAR scan and returns the flattened 3D point cloud (x, y, z) float array
        /// of length TotalBeamCount * 3 for the ML-Agents VectorSensor.
        /// </summary>
        public float[] Perceive()
        {
            if (!isInitialized)
            {
                Initialize();
            }

            int expectedBeams = TotalBeamCount;
            int expectedObsSize = ObservationSize;
            if (normalizedRanges == null || normalizedRanges.Length != expectedBeams ||
                observationBuffer == null || observationBuffer.Length != expectedObsSize)
            {
                AllocateBuffers();
            }

            bool shouldUpdateScan = rotationFrequencyHz <= 0f ||
                                    lastScanTime < -100f ||
                                    (Time.time - lastScanTime) >= (1f / rotationFrequencyHz);

            if (shouldUpdateScan)
            {
                CaptureScan();
                lastScanTime = Time.time;
            }

            return observationBuffer;
        }

        private Camera[] FindRigCameras()
        {
            Transform selfDrivingCamsTransform = transform.Find("Self_driving_cameras");
            if (selfDrivingCamsTransform != null)
            {
                if (!selfDrivingCamsTransform.gameObject.activeSelf)
                {
                    selfDrivingCamsTransform.gameObject.SetActive(true);
                }

                Camera[] childCams = selfDrivingCamsTransform.GetComponentsInChildren<Camera>(true);
                if (childCams != null && childCams.Length > 0)
                {
                    return SortCamerasByAzimuth(childCams);
                }
            }

            Camera[] allChildCams = GetComponentsInChildren<Camera>(true);
            return SortCamerasByAzimuth(allChildCams);
        }

        private Camera[] SortCamerasByAzimuth(Camera[] cams)
        {
            if (cams == null || cams.Length <= 1)
            {
                return cams;
            }

            List<Camera> validCams = new List<Camera>(cams.Length);
            for (int i = 0; i < cams.Length; i++)
            {
                if (cams[i] != null)
                {
                    validCams.Add(cams[i]);
                }
            }

            validCams.Sort((a, b) =>
            {
                Vector3 fwdA = transform.InverseTransformDirection(a.transform.forward);
                Vector3 fwdB = transform.InverseTransformDirection(b.transform.forward);
                float angleA = Mathf.Atan2(fwdA.x, fwdA.z) * Mathf.Rad2Deg;
                float angleB = Mathf.Atan2(fwdB.x, fwdB.z) * Mathf.Rad2Deg;
                return angleA.CompareTo(angleB);
            });

            return validCams.ToArray();
        }

        private void ActivateCameraHierarchy()
        {
            if (lidarCameras == null)
            {
                return;
            }

            foreach (Camera cam in lidarCameras)
            {
                if (cam == null)
                {
                    continue;
                }

                Transform current = cam.transform;
                while (current != null && current != transform)
                {
                    if (!current.gameObject.activeSelf)
                    {
                        current.gameObject.SetActive(true);
                    }
                    current = current.parent;
                }

                cam.enabled = false;
            }

            self_driving_camera_manager fovManager = GetComponentInChildren<self_driving_camera_manager>(true);
            if (fovManager != null)
            {
                fovManager.enabled = false;
            }
        }

        private void NormalizeNegativeScaleContainers()
        {
            if (lidarCameras == null)
            {
                return;
            }

            HashSet<Transform> parents = new HashSet<Transform>();
            foreach (Camera cam in lidarCameras)
            {
                if (cam != null && cam.transform.parent != null && cam.transform.parent != transform)
                {
                    parents.Add(cam.transform.parent);
                }
            }

            foreach (Transform parent in parents)
            {
                Vector3 scale = parent.localScale;
                if (scale.x < 0f)
                {
                    int childCount = parent.childCount;
                    Vector3[] localPositions = new Vector3[childCount];
                    Quaternion[] localRotations = new Quaternion[childCount];

                    for (int i = 0; i < childCount; i++)
                    {
                        Transform child = parent.GetChild(i);
                        Vector3 lp = child.localPosition;
                        localPositions[i] = new Vector3(-lp.x, lp.y, lp.z);

                        Vector3 euler = child.localEulerAngles;
                        localRotations[i] = Quaternion.Euler(euler.x, -euler.y, -euler.z);
                    }

                    parent.localScale = new Vector3(Mathf.Abs(scale.x), scale.y, scale.z);

                    for (int i = 0; i < childCount; i++)
                    {
                        Transform child = parent.GetChild(i);
                        child.localPosition = localPositions[i];
                        child.localRotation = localRotations[i];
                    }
                }
            }

            lidarCameras = SortCamerasByAzimuth(lidarCameras);
        }

        private void ConfigureCameras()
        {
            if (lidarCameras == null || lidarCameras.Length == 0)
            {
                return;
            }

            horizontalSamplesPerCamera = Mathf.Max(1, horizontalSamplesPerCamera);
            verticalChannels = Mathf.Max(2, verticalChannels);
            minRange = Mathf.Max(0.05f, minRange);
            maxRange = Mathf.Max(minRange + 0.1f, maxRange);

            float verticalFov = Mathf.Clamp(maxVerticalAngle - minVerticalAngle, 1f, 150f);
            float pitchCenterDeg = 0.5f * (minVerticalAngle + maxVerticalAngle);
            float halfHFovRad = horizontalFov * 0.5f * Mathf.Deg2Rad;
            float halfVFovRad = verticalFov * 0.5f * Mathf.Deg2Rad;
            float aspect = Mathf.Tan(halfHFovRad) / Mathf.Max(0.001f, Mathf.Tan(halfVFovRad));

            int camCount = lidarCameras.Length;
            float sliceWidth = 1f / camCount;

            for (int i = 0; i < camCount; i++)
            {
                Camera cam = lidarCameras[i];
                if (cam == null)
                {
                    continue;
                }

                Vector3 localEuler = cam.transform.localEulerAngles;
                cam.transform.localRotation = Quaternion.Euler(-pitchCenterDeg, localEuler.y, 0f);

                cam.enabled = false;
                cam.orthographic = false;
                cam.usePhysicalProperties = false;
                // Each camera renders into its own horizontal slice of the shared 3D LiDAR atlas RenderTexture
                cam.rect = new Rect(i * sliceWidth, 0f, sliceWidth, 1f);
                cam.fieldOfView = verticalFov;
                cam.aspect = aspect;
                cam.nearClipPlane = minRange;
                cam.farClipPlane = maxRange;
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = Color.white;
                cam.cullingMask = cullingMask;
                cam.allowHDR = false;
                cam.allowMSAA = false;
                cam.useOcclusionCulling = false;
                cam.depthTextureMode = DepthTextureMode.None;
                cam.ResetProjectionMatrix();
            }
        }

        private void AllocateBuffers()
        {
            int camCount = Mathf.Max(1, CameraCount);
            int singleWidth = Mathf.Max(1, horizontalSamplesPerCamera);
            int atlasWidth = camCount * singleWidth;
            int height = Mathf.Max(2, verticalChannels);
            int totalBeams = TotalBeamCount;
            int obsSize = ObservationSize;

            normalizedRanges = new float[totalBeams];
            observationBuffer = new float[obsSize];
            localPointCloudXYZ = new Vector3[totalBeams];
            worldPointCloudXYZ = new Vector3[totalBeams];
            lastWorldRayOrigins = new Vector3[totalBeams];
            lastWorldRayDirections = new Vector3[totalBeams];
            cachedLocalRayOrigins = new Vector3[totalBeams];
            cachedLocalRayDirections = new Vector3[totalBeams];
            lastHitDistances = new float[totalBeams];

            for (int i = 0; i < totalBeams; i++)
            {
                normalizedRanges[i] = 1f;
                lastHitDistances[i] = maxRange;
            }

            PrecomputeLocalRays();
            ReleaseTextures();

            useRFloatTexture = SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat);
            RenderTextureFormat rtFormat = useRFloatTexture ? RenderTextureFormat.RFloat : RenderTextureFormat.ARGB32;
            TextureFormat texFormat = useRFloatTexture ? TextureFormat.RFloat : TextureFormat.RGBA32;

            scanRenderTexture = new RenderTexture(atlasWidth, height, 16, rtFormat, RenderTextureReadWrite.Linear)
            {
                name = "Lidar3DScanAtlasRT",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                antiAliasing = 1
            };
            scanRenderTexture.Create();

            readbackTexture = new Texture2D(atlasWidth, height, texFormat, false, true)
            {
                name = "Lidar3DAtlasReadbackTex",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
        }

        private void PrecomputeLocalRays()
        {
            if (lidarCameras == null || lidarCameras.Length == 0)
            {
                return;
            }

            int width = Mathf.Max(1, horizontalSamplesPerCamera);
            int height = Mathf.Max(2, verticalChannels);
            int beamIndex = 0;

            for (int camIdx = 0; camIdx < lidarCameras.Length; camIdx++)
            {
                Camera cam = lidarCameras[camIdx];
                if (cam == null)
                {
                    continue;
                }

                for (int v = 0; v < height; v++)
                {
                    float viewportV = (v + 0.5f) / height;
                    for (int u = 0; u < width; u++)
                    {
                        float viewportU = width == 1 ? 0.5f : (u + 0.5f) / width;
                        Ray worldRay = GetBeamWorldRay(cam, viewportU, viewportV);
                        cachedLocalRayOrigins[beamIndex] = transform.InverseTransformPoint(worldRay.origin);
                        cachedLocalRayDirections[beamIndex] = transform.InverseTransformDirection(worldRay.direction).normalized;
                        beamIndex++;
                    }
                }
            }
        }

        private void CaptureScan()
        {
            if (lidarCameras == null || lidarCameras.Length == 0)
            {
                return;
            }

            // During Python ML-Agents training (Communicator on), bypass GPU Camera.RenderWithShader + synchronous ReadPixels
            // to eliminate GPU-CPU pipeline stalls and maximize training iterations/sec.
            bool bypassGpuReadbackForTraining = autoDisableVisualsDuringTraining &&
                                                Academy.IsInitialized &&
                                                Academy.Instance.IsCommunicatorOn;

            bool canUseGpuShader = !bypassGpuReadbackForTraining &&
                                   samplingMode == LidarSamplingMode.CameraDepthShader &&
                                   SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null &&
                                   lidarDepthShader != null &&
                                   lidarDepthShader.isSupported;

            if (canUseGpuShader)
            {
                CaptureScanWithCameraShader();
            }
            else
            {
                CaptureScanWithPhysicsRaycast();
            }

            PopulateObservationBuffer();
        }

        private void CaptureScanWithCameraShader()
        {
            int camCount = lidarCameras.Length;
            int singleWidth = Mathf.Max(1, horizontalSamplesPerCamera);
            int atlasWidth = camCount * singleWidth;
            int height = Mathf.Max(2, verticalChannels);

            if (scanRenderTexture == null || scanRenderTexture.width != atlasWidth || scanRenderTexture.height != height)
            {
                ConfigureCameras();
                AllocateBuffers();
            }

            Shader.SetGlobalFloat(LidarMaxRangeShaderId, maxRange);
            RenderTexture previousActive = RenderTexture.active;

            SetIgnoredRenderersEnabled(false);
            try
            {
                // Render all 8 cameras into their respective slices of scanRenderTexture
                for (int camIdx = 0; camIdx < camCount; camIdx++)
                {
                    Camera cam = lidarCameras[camIdx];
                    if (cam == null)
                    {
                        continue;
                    }

                    RenderTexture prevCamTarget = cam.targetTexture;
                    cam.targetTexture = scanRenderTexture;
                    cam.RenderWithShader(lidarDepthShader, "");
                    cam.targetTexture = prevCamTarget;
                }
            }
            finally
            {
                SetIgnoredRenderersEnabled(true);
            }

            // Single GPU-to-CPU ReadPixels call for the entire 8-camera 3D atlas
            RenderTexture.active = scanRenderTexture;
            readbackTexture.ReadPixels(new Rect(0, 0, atlasWidth, height), 0, 0, false);
            RenderTexture.active = previousActive;

            Color[] pixels = readbackTexture.GetPixels();

            int beamIndex = 0;

            for (int camIdx = 0; camIdx < camCount; camIdx++)
            {
                Camera cam = lidarCameras[camIdx];
                if (cam == null)
                {
                    continue;
                }

                int xOffset = camIdx * singleWidth;

                for (int v = 0; v < height; v++)
                {
                    int rowOffset = v * atlasWidth + xOffset;

                    for (int u = 0; u < singleWidth; u++)
                    {
                        Color pixel = pixels[rowOffset + u];

                        float normDist = Mathf.Clamp01(pixel.r);
                        float distanceMeters = normDist * maxRange;

                        if (rangeNoiseStdDev > 0f && normDist < 0.999f)
                        {
                            distanceMeters = Mathf.Clamp(distanceMeters + SampleGaussianNoise(rangeNoiseStdDev), minRange, maxRange);
                            normDist = distanceMeters / maxRange;
                        }

                        Vector3 localOrigin = cachedLocalRayOrigins[beamIndex];
                        Vector3 localDir = cachedLocalRayDirections[beamIndex];
                        Vector3 localHit = localOrigin + localDir * distanceMeters;
                        Vector3 worldOrigin = transform.TransformPoint(localOrigin);
                        Vector3 worldDir = transform.TransformDirection(localDir);
                        Vector3 worldHit = worldOrigin + worldDir * distanceMeters;

                        if (ignoreGroundAndTriggers && normDist < 0.999f && localHit.y <= groundHeightThreshold)
                        {
                            normDist = 1f;
                            distanceMeters = maxRange;
                            worldHit = worldOrigin + worldDir * distanceMeters;
                            localHit = Vector3.zero;
                        }

                        normalizedRanges[beamIndex] = normDist;
                        lastWorldRayOrigins[beamIndex] = worldOrigin;
                        lastWorldRayDirections[beamIndex] = worldDir;
                        lastHitDistances[beamIndex] = distanceMeters;
                        worldPointCloudXYZ[beamIndex] = worldHit;
                        localPointCloudXYZ[beamIndex] = normDist < 0.999f ? localHit : Vector3.zero;

                        beamIndex++;
                    }
                }
            }
        }

        private void CaptureScanWithPhysicsRaycast()
        {
            int width = Mathf.Max(1, horizontalSamplesPerCamera);
            int height = Mathf.Max(2, verticalChannels);
            int camCount = lidarCameras.Length;
            int totalBeams = camCount * width * height;

            if (cachedLocalRayDirections == null || cachedLocalRayDirections.Length != totalBeams)
            {
                PrecomputeLocalRays();
            }

            int beamIndex = 0;

            SetEgoCollidersEnabled(false);
            try
            {
                for (int camIdx = 0; camIdx < camCount; camIdx++)
                {
                    if (lidarCameras[camIdx] == null)
                    {
                        continue;
                    }

                    for (int v = 0; v < height; v++)
                    {
                        for (int u = 0; u < width; u++)
                        {
                            Vector3 localOrigin = cachedLocalRayOrigins[beamIndex];
                            Vector3 localDir = cachedLocalRayDirections[beamIndex];
                            Vector3 worldOrigin = transform.TransformPoint(localOrigin);
                            Vector3 worldDir = transform.TransformDirection(localDir);

                            float distanceMeters = maxRange;
                            float normDist = 1f;
                            Vector3 worldHit = worldOrigin + worldDir * maxRange;
                            Vector3 localHit = Vector3.zero;

                            if (Physics.Raycast(worldOrigin, worldDir, out RaycastHit hit, maxRange, cullingMask, QueryTriggerInteraction.Ignore))
                            {
                                Vector3 candidateLocalHit = localOrigin + localDir * hit.distance;
                                bool isGroundHit = ignoreGroundAndTriggers &&
                                                   (candidateLocalHit.y <= groundHeightThreshold ||
                                                    hit.collider.gameObject.name == "Plane" ||
                                                    hit.collider.CompareTag("Finish"));

                                if (!isGroundHit)
                                {
                                    distanceMeters = Mathf.Clamp(hit.distance, minRange, maxRange);
                                    if (rangeNoiseStdDev > 0f)
                                    {
                                        distanceMeters = Mathf.Clamp(distanceMeters + SampleGaussianNoise(rangeNoiseStdDev), minRange, maxRange);
                                    }
                                    normDist = distanceMeters / maxRange;
                                    localHit = localOrigin + localDir * distanceMeters;
                                    worldHit = worldOrigin + worldDir * distanceMeters;
                                }
                            }

                            normalizedRanges[beamIndex] = normDist;
                            lastWorldRayOrigins[beamIndex] = worldOrigin;
                            lastWorldRayDirections[beamIndex] = worldDir;
                            lastHitDistances[beamIndex] = distanceMeters;
                            worldPointCloudXYZ[beamIndex] = worldHit;
                            localPointCloudXYZ[beamIndex] = localHit;

                            beamIndex++;
                        }
                    }
                }
            }
            finally
            {
                SetEgoCollidersEnabled(true);
            }
        }

        /// <summary>
        /// Populates observationBuffer with 3D (x, y, z) coordinates for the ML-Agents VectorSensor.
        /// When poolVerticalChannelsForMlp is true, pools across the vertical elevation channels for each
        /// horizontal azimuth column by selecting the closest 3D hit point (x, y, z), giving native SB3 MlpPolicy
        /// a compact 3D point cloud representation (128x3 = 384 floats instead of 512x3 = 1536).
        /// </summary>
        private void PopulateObservationBuffer()
        {
            int camCount = lidarCameras != null ? lidarCameras.Length : 0;
            int width = Mathf.Max(1, horizontalSamplesPerCamera);
            int height = Mathf.Max(2, verticalChannels);
            int obsIndex = 0;

            if (!poolVerticalChannelsForMlp)
            {
                int total = TotalBeamCount;
                for (int i = 0; i < total; i++)
                {
                    Write3DPointObservation(localPointCloudXYZ[i], normalizedRanges[i], ref obsIndex);
                }
                return;
            }

            int camStride = width * height;
            for (int camIdx = 0; camIdx < camCount; camIdx++)
            {
                int camBase = camIdx * camStride;
                for (int u = 0; u < width; u++)
                {
                    float closestNormDist = 1f;
                    Vector3 closestLocalHit = Vector3.zero;

                    for (int v = 0; v < height; v++)
                    {
                        int beamIdx = camBase + v * width + u;
                        float normDist = normalizedRanges[beamIdx];
                        if (normDist < closestNormDist)
                        {
                            closestNormDist = normDist;
                            closestLocalHit = localPointCloudXYZ[beamIdx];
                        }
                    }

                    Write3DPointObservation(closestLocalHit, closestNormDist, ref obsIndex);
                }
            }
        }

        private Ray GetBeamWorldRay(Camera cam, float viewportU, float viewportV)
        {
            float verticalFov = Mathf.Clamp(maxVerticalAngle - minVerticalAngle, 1f, 150f);
            float tanHalfHFov = Mathf.Tan(horizontalFov * 0.5f * Mathf.Deg2Rad);
            float tanHalfVFov = Mathf.Tan(verticalFov * 0.5f * Mathf.Deg2Rad);

            float ndcX = viewportU * 2f - 1f;
            float ndcY = viewportV * 2f - 1f;

            Vector3 localDir = new Vector3(ndcX * tanHalfHFov, ndcY * tanHalfVFov, 1f).normalized;
            Vector3 worldDir = cam.transform.TransformDirection(localDir);
            return new Ray(cam.transform.position, worldDir.normalized);
        }

        private void Write3DPointObservation(Vector3 localHitMeters, float normDist, ref int obsIndex)
        {
            if (normDist >= 0.999f)
            {
                observationBuffer[obsIndex++] = 0f;
                observationBuffer[obsIndex++] = 0f;
                observationBuffer[obsIndex++] = 0f;
            }
            else
            {
                Vector3 normalizedXYZ = localHitMeters / maxRange;
                observationBuffer[obsIndex++] = Mathf.Clamp(normalizedXYZ.x, -1f, 1f);
                observationBuffer[obsIndex++] = Mathf.Clamp(normalizedXYZ.y, -1f, 1f);
                observationBuffer[obsIndex++] = Mathf.Clamp(normalizedXYZ.z, -1f, 1f);
            }
        }

        private void CacheIgnoredSceneRenderers()
        {
            ignoredSceneRenderers.Clear();
            ignoredSceneRendererStates.Clear();

            Renderer[] allRenderers = FindObjectsByType<Renderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int i = 0; i < allRenderers.Length; i++)
            {
                Renderer r = allRenderers[i];
                if (r == null || r.transform.IsChildOf(transform))
                {
                    continue;
                }

                GameObject go = r.gameObject;
                Collider col = go.GetComponent<Collider>();
                bool isTriggerVisual = (col != null && col.isTrigger) || go.CompareTag("Finish");
                bool isGroundPlane = go.name == "Plane";

                if (isTriggerVisual || isGroundPlane)
                {
                    ignoredSceneRenderers.Add(r);
                    ignoredSceneRendererStates.Add(r.enabled);
                }
            }
        }

        private void SetIgnoredRenderersEnabled(bool enabledState)
        {
            if (egoRenderers != null)
            {
                for (int i = 0; i < egoRenderers.Length; i++)
                {
                    Renderer r = egoRenderers[i];
                    if (r == null)
                    {
                        continue;
                    }

                    if (!enabledState)
                    {
                        egoRendererStates[i] = r.enabled;
                        r.enabled = false;
                    }
                    else
                    {
                        r.enabled = egoRendererStates[i];
                    }
                }
            }

            if (!ignoreGroundAndTriggers)
            {
                return;
            }

            for (int i = 0; i < ignoredSceneRenderers.Count; i++)
            {
                Renderer r = ignoredSceneRenderers[i];
                if (r == null)
                {
                    continue;
                }

                if (!enabledState)
                {
                    ignoredSceneRendererStates[i] = r.enabled;
                    r.enabled = false;
                }
                else
                {
                    r.enabled = ignoredSceneRendererStates[i];
                }
            }
        }

        private void SetEgoCollidersEnabled(bool enabledState)
        {
            if (egoColliders == null)
            {
                return;
            }

            for (int i = 0; i < egoColliders.Length; i++)
            {
                Collider c = egoColliders[i];
                if (c == null)
                {
                    continue;
                }

                if (!enabledState)
                {
                    egoColliderStates[i] = c.enabled;
                    c.enabled = false;
                }
                else
                {
                    c.enabled = egoColliderStates[i];
                }
            }
        }

        private static float SampleGaussianNoise(float stdDev)
        {
            float u1 = 1.0f - UnityEngine.Random.value;
            float u2 = 1.0f - UnityEngine.Random.value;
            float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(Mathf.Max(u1, 1e-6f))) * Mathf.Sin(2.0f * Mathf.PI * u2);
            return stdDev * randStdNormal;
        }

        private void ReleaseTextures()
        {
            if (scanRenderTexture != null)
            {
                scanRenderTexture.Release();
                Destroy(scanRenderTexture);
                scanRenderTexture = null;
            }

            if (readbackTexture != null)
            {
                Destroy(readbackTexture);
                readbackTexture = null;
            }
        }

        private void OnEnable()
        {
            if (primaryVisualSensor == null)
            {
                primaryVisualSensor = this;
            }
        }

        private void OnDisable()
        {
            if (primaryVisualSensor == this)
            {
                primaryVisualSensor = null;
            }
        }

        private static Texture2D viewerWhiteTex;
        private float viewerYawDeg = 35f;
        private float viewerPitchDeg = 28f;
        private float viewerDistance = 16f;
        private bool isDraggingViewer = false;

        private struct Projected3DPoint : IComparable<Projected3DPoint>
        {
            public Vector2 screenPos;
            public Vector2 groundScreenPos;
            public bool hasGroundPos;
            public float depthZ;
            public float sizePx;
            public Color color;

            public int CompareTo(Projected3DPoint other)
            {
                return other.depthZ.CompareTo(depthZ); // Back-to-front painter's sort
            }
        }

        private readonly List<Projected3DPoint> projectedPointsBuffer = new List<Projected3DPoint>(1024);

        private bool ShouldRenderVisuals()
        {
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
            {
                return false;
            }

            if (autoDisableVisualsDuringTraining && Academy.IsInitialized && Academy.Instance.IsCommunicatorOn)
            {
                return false;
            }

            return true;
        }

        private void LateUpdate()
        {
            if (!isInitialized || primaryVisualSensor != this || !ShouldRenderVisuals())
            {
                return;
            }

            // Only trigger a fallback Perceive() from LateUpdate when CollectObservations() is not actively polling
            // (e.g., during the heuristic spot-finding cruise phase before RequestDecision() begins),
            // so we never alter the deterministic FixedUpdate scan timing during RL steps.
            if (render3DPointCloud || show3DPointCloudViewer || drawDebugRays)
            {
                float interval = rotationFrequencyHz > 0f ? (1f / rotationFrequencyHz) : 0.1f;
                float staleThreshold = Mathf.Max(interval * 2.5f, 0.25f);
                if (Time.time - lastScanTime >= staleThreshold)
                {
                    Perceive();
                }
            }

            if (render3DPointCloud)
            {
                Draw3DPointCloudMeshes();
            }
        }

        private void Ensure3DPointResources()
        {
            if (sharedSphereMesh == null)
            {
                // Load built-in sphere mesh directly without spawning a temporary SphereCollider in the physics world
                sharedSphereMesh = Resources.GetBuiltinResource<Mesh>("New-Sphere.fbx");
                if (sharedSphereMesh == null)
                {
                    sharedSphereMesh = Resources.GetBuiltinResource<Mesh>("Sphere.fbx");
                }
                if (sharedSphereMesh == null)
                {
                    GameObject tempSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    Collider tempCol = tempSphere.GetComponent<Collider>();
                    if (tempCol != null)
                    {
                        tempCol.enabled = false;
                    }
                    MeshFilter mf = tempSphere.GetComponent<MeshFilter>();
                    if (mf != null)
                    {
                        sharedSphereMesh = mf.sharedMesh;
                    }
                    DestroyImmediate(tempSphere);
                }
            }

            if (sharedPointMaterial == null)
            {
                // Sprites/Default supports per-instance _Color via MaterialPropertyBlock in Built-in RP
                // and has no "RenderType"="Opaque" tag, so replacement shaders ignore it
                Shader unlitShader = Shader.Find("Sprites/Default");
                if (unlitShader == null)
                {
                    unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
                }
                if (unlitShader == null)
                {
                    unlitShader = Shader.Find("Unlit/Color");
                }
                sharedPointMaterial = new Material(unlitShader)
                {
                    name = "Lidar3DPointCloudMat",
                    enableInstancing = true
                };
            }

            if (pointPropertyBlock == null)
            {
                pointPropertyBlock = new MaterialPropertyBlock();
            }
        }

        private void Draw3DPointCloudMeshes()
        {
            if (worldPointCloudXYZ == null || normalizedRanges == null)
            {
                return;
            }

            // Render 3D point cloud spheres ONLY to Camera.main so they are NEVER rendered into any LiDAR depth camera
            Camera displayCam = Camera.main;
            if (displayCam == null)
            {
                return;
            }

            Ensure3DPointResources();
            if (sharedSphereMesh == null || sharedPointMaterial == null)
            {
                return;
            }

            float diameter = pointCloudSphereRadius * 2f;
            Vector3 scale = new Vector3(diameter, diameter, diameter);
            int count = Mathf.Min(worldPointCloudXYZ.Length, normalizedRanges.Length);

            for (int i = 0; i < count; i++)
            {
                float norm = normalizedRanges[i];
                if (norm >= 0.999f)
                {
                    continue;
                }

                Color hitColor = Color.HSVToRGB(Mathf.Clamp01(norm * 0.35f), 0.95f, 1f);
                pointPropertyBlock.SetColor(BaseColorShaderId, hitColor);
                pointPropertyBlock.SetColor(ColorShaderId, hitColor);

                Matrix4x4 matrix = Matrix4x4.TRS(worldPointCloudXYZ[i], Quaternion.identity, scale);
                Graphics.DrawMesh(sharedSphereMesh, matrix, sharedPointMaterial, 0, displayCam, 0, pointPropertyBlock, ShadowCastingMode.Off, false);
            }
        }

        private void OnDestroy()
        {
            if (primaryVisualSensor == this)
            {
                primaryVisualSensor = null;
            }
            ReleaseTextures();
        }

        private void OnDrawGizmos()
        {
            if (!ShouldRenderVisuals() ||
                (!render3DPointCloud && !drawDebugRays) ||
                lastWorldRayOrigins == null || lastWorldRayDirections == null || normalizedRanges == null)
            {
                return;
            }

            if (primaryVisualSensor != null && primaryVisualSensor != this)
            {
                return;
            }

            int count = Mathf.Min(lastWorldRayOrigins.Length, normalizedRanges.Length);
            for (int i = 0; i < count; i++)
            {
                float norm = normalizedRanges[i];
                float dist = lastHitDistances != null && i < lastHitDistances.Length ? lastHitDistances[i] : norm * maxRange;
                Vector3 origin = lastWorldRayOrigins[i];
                Vector3 end = origin + lastWorldRayDirections[i] * dist;

                if (norm < 0.999f)
                {
                    Color hitColor = Color.HSVToRGB(Mathf.Clamp01(norm * 0.35f), 0.95f, 1f);
                    Gizmos.color = hitColor;

                    if (render3DPointCloud)
                    {
                        Gizmos.DrawSphere(end, pointCloudSphereRadius);
                    }
                    if (drawDebugRays)
                    {
                        Gizmos.color = new Color(hitColor.r, hitColor.g, hitColor.b, 0.35f);
                        Gizmos.DrawLine(origin, end);
                    }
                }
                else if (drawDebugRays)
                {
                    Gizmos.color = new Color(0f, 1f, 0f, 0.12f);
                    Gizmos.DrawLine(origin, end);
                }
            }
        }

        private void OnGUI()
        {
            if (!ShouldRenderVisuals() || !show3DPointCloudViewer || primaryVisualSensor != this || localPointCloudXYZ == null || normalizedRanges == null)
            {
                return;
            }

            if (viewerWhiteTex == null)
            {
                viewerWhiteTex = new Texture2D(1, 1);
                viewerWhiteTex.SetPixel(0, 0, Color.white);
                viewerWhiteTex.Apply();
            }

            const float margin = 12f;
            const float viewWidth = 310f;
            const float viewHeight = 250f;
            const float footerHeight = 46f;
            Rect panelRect = new Rect(margin, margin, viewWidth + 16f, viewHeight + footerHeight + 32f);
            Rect viewportRect = new Rect(panelRect.x + 8f, panelRect.y + 24f, viewWidth, viewHeight);

            Handle3DViewerInput(viewportRect);

            // Outer panel background
            GUI.color = new Color(0.07f, 0.08f, 0.11f, 0.90f);
            GUI.DrawTexture(panelRect, viewerWhiteTex);

            // Header title
            GUI.color = Color.white;
            GUIStyle headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleLeft
            };
            GUI.Label(new Rect(panelRect.x + 10f, panelRect.y + 4f, viewWidth - 60f, 18f),
                $"3D LiDAR Point Cloud ({TotalBeamCount} XYZ pts)", headerStyle);

            if (GUI.Button(new Rect(panelRect.xMax - 64f, panelRect.y + 3f, 56f, 18f), "Reset 3D"))
            {
                viewerYawDeg = 35f;
                viewerPitchDeg = 28f;
                viewerDistance = 16f;
            }

            // 3D Viewport background
            GUI.color = new Color(0.02f, 0.03f, 0.05f, 0.96f);
            GUI.DrawTexture(viewportRect, viewerWhiteTex);

            Quaternion camRot = Quaternion.Euler(viewerPitchDeg, viewerYawDeg, 0f);
            Vector3 camTarget = new Vector3(0f, 0.6f, 0f);
            Vector3 camPos = camTarget - (camRot * Vector3.forward) * viewerDistance;
            float focalLength = viewHeight * 0.95f;

            GUI.BeginClip(viewportRect);
            Rect localViewport = new Rect(0f, 0f, viewWidth, viewHeight);

            // 1. Draw 3D ground reference grid in ego-car 3D space (y = 0)
            Color gridColor = new Color(0.16f, 0.22f, 0.30f, 0.55f);
            for (int g = -12; g <= 12; g += 4)
            {
                Draw3DLineInViewport(new Vector3(g, 0f, -12f), new Vector3(g, 0f, 12f), camPos, camRot, focalLength, localViewport, gridColor, 1f);
                Draw3DLineInViewport(new Vector3(-12f, 0f, g), new Vector3(12f, 0f, g), camPos, camRot, focalLength, localViewport, gridColor, 1f);
            }

            // 2. Draw 3D Ego Car Wireframe Box (1.8m W x 1.4m H x 4.2m L) and 3D XYZ axes
            Draw3DWireBox(new Vector3(0f, 0.7f, 0f), new Vector3(1.8f, 1.4f, 4.2f), camPos, camRot, focalLength, localViewport, new Color(0.2f, 0.65f, 1f, 0.75f));
            Draw3DLineInViewport(Vector3.zero, new Vector3(2.2f, 0f, 0f), camPos, camRot, focalLength, localViewport, new Color(1f, 0.25f, 0.25f, 0.9f), 2f); // +X Right
            Draw3DLineInViewport(Vector3.zero, new Vector3(0f, 2.2f, 0f), camPos, camRot, focalLength, localViewport, new Color(0.25f, 1f, 0.35f, 0.9f), 2f); // +Y Up
            Draw3DLineInViewport(Vector3.zero, new Vector3(0f, 0f, 3.0f), camPos, camRot, focalLength, localViewport, new Color(0.3f, 0.6f, 1f, 0.9f), 2f);   // +Z Forward

            // 3. Project and depth-sort all active 3D Point Cloud (x, y, z) coordinates
            projectedPointsBuffer.Clear();
            int count = Mathf.Min(localPointCloudXYZ.Length, normalizedRanges.Length);
            int activeHitCount = 0;
            Vector3 sampleHitXYZ = Vector3.zero;

            for (int i = 0; i < count; i++)
            {
                float norm = normalizedRanges[i];
                if (norm >= 0.999f)
                {
                    continue;
                }

                Vector3 ptXYZ = localPointCloudXYZ[i];
                activeHitCount++;
                if (activeHitCount == 1)
                {
                    sampleHitXYZ = ptXYZ;
                }

                if (TryProject3DPoint(ptXYZ, camPos, camRot, focalLength, localViewport, out Vector2 screenPt, out float depthZ))
                {
                    Vector2 groundScreenPt = Vector2.zero;
                    bool hasGround = show3DElevationStalks &&
                                     TryProject3DPoint(new Vector3(ptXYZ.x, 0f, ptXYZ.z), camPos, camRot, focalLength, localViewport, out groundScreenPt, out _);

                    float pointSize = Mathf.Clamp(52f / Mathf.Max(1f, depthZ), 2.5f, 6.5f);
                    Color hitColor = Color.HSVToRGB(Mathf.Clamp01(norm * 0.35f), 0.95f, 1f);

                    projectedPointsBuffer.Add(new Projected3DPoint
                    {
                        screenPos = screenPt,
                        groundScreenPos = groundScreenPt,
                        hasGroundPos = hasGround,
                        depthZ = depthZ,
                        sizePx = pointSize,
                        color = hitColor
                    });
                }
            }

            projectedPointsBuffer.Sort();

            // 4. Render 3D elevation stalks and 3D point cloud spheres
            for (int i = 0; i < projectedPointsBuffer.Count; i++)
            {
                Projected3DPoint p = projectedPointsBuffer[i];
                if (p.hasGroundPos)
                {
                    Color stalkColor = new Color(p.color.r, p.color.g, p.color.b, 0.28f);
                    Draw2DLineClipped(p.groundScreenPos, p.screenPos, stalkColor, 1f);
                }

                GUI.color = p.color;
                float half = p.sizePx * 0.5f;
                GUI.DrawTexture(new Rect(p.screenPos.x - half, p.screenPos.y - half, p.sizePx, p.sizePx), viewerWhiteTex);
            }

            GUI.color = new Color(0.75f, 0.82f, 0.92f, 0.75f);
            GUI.Label(new Rect(6f, viewHeight - 18f, viewWidth - 12f, 16f),
                $"Drag mouse to orbit 3D | Scroll to zoom | Layers: {verticalChannels}");

            GUI.EndClip();

            // Footer telemetry showing 3D (x, y, z) observation vector stats
            GUIStyle footerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                alignment = TextAnchor.UpperLeft
            };
            GUI.color = Color.white;
            Rect footerRect = new Rect(panelRect.x + 8f, viewportRect.yMax + 4f, viewWidth, footerHeight);
            GUI.Label(footerRect,
                $"3D Hits: {activeHitCount}/{TotalBeamCount}  |  VectorSensor: {ObservationSize} floats ({ObservationPointCount}×XYZ)\n" +
                $"Sample 3D Point (m): X={sampleHitXYZ.x:F2}, Y={sampleHitXYZ.y:F2}, Z={sampleHitXYZ.z:F2}",
                footerStyle);
        }

        private void Handle3DViewerInput(Rect viewportRect)
        {
            Event e = Event.current;
            if (e == null)
            {
                return;
            }

            if (e.type == EventType.MouseDown && e.button == 0 && viewportRect.Contains(e.mousePosition))
            {
                isDraggingViewer = true;
                e.Use();
            }
            else if (e.type == EventType.MouseUp && e.button == 0)
            {
                isDraggingViewer = false;
            }
            else if (e.type == EventType.MouseDrag && isDraggingViewer)
            {
                viewerYawDeg += e.delta.x * 0.6f;
                viewerPitchDeg = Mathf.Clamp(viewerPitchDeg + e.delta.y * 0.5f, 5f, 85f);
                e.Use();
            }
            else if (e.type == EventType.ScrollWheel && viewportRect.Contains(e.mousePosition))
            {
                viewerDistance = Mathf.Clamp(viewerDistance + e.delta.y * 0.8f, 5f, 35f);
                e.Use();
            }
        }

        private static bool TryProject3DPoint(Vector3 localPoint, Vector3 camPos, Quaternion camRot, float focalLength, Rect viewport, out Vector2 screenPos, out float depthZ)
        {
            Vector3 viewSpace = Quaternion.Inverse(camRot) * (localPoint - camPos);
            depthZ = viewSpace.z;
            if (depthZ <= 0.2f)
            {
                screenPos = Vector2.zero;
                return false;
            }

            float sx = viewport.width * 0.5f + (viewSpace.x / depthZ) * focalLength;
            float sy = viewport.height * 0.5f - (viewSpace.y / depthZ) * focalLength;
            screenPos = new Vector2(sx, sy);
            return sx >= -20f && sx <= viewport.width + 20f && sy >= -20f && sy <= viewport.height + 20f;
        }

        private static void Draw3DWireBox(Vector3 center, Vector3 size, Vector3 camPos, Quaternion camRot, float focal, Rect viewport, Color color)
        {
            Vector3 h = size * 0.5f;
            Vector3[] c = new Vector3[8]
            {
                center + new Vector3(-h.x, -h.y, -h.z),
                center + new Vector3( h.x, -h.y, -h.z),
                center + new Vector3( h.x, -h.y,  h.z),
                center + new Vector3(-h.x, -h.y,  h.z),
                center + new Vector3(-h.x,  h.y, -h.z),
                center + new Vector3( h.x,  h.y, -h.z),
                center + new Vector3( h.x,  h.y,  h.z),
                center + new Vector3(-h.x,  h.y,  h.z)
            };

            for (int i = 0; i < 4; i++)
            {
                int next = (i + 1) % 4;
                Draw3DLineInViewport(c[i], c[next], camPos, camRot, focal, viewport, color, 1.5f);
                Draw3DLineInViewport(c[i + 4], c[next + 4], camPos, camRot, focal, viewport, color, 1.5f);
                Draw3DLineInViewport(c[i], c[i + 4], camPos, camRot, focal, viewport, color, 1.5f);
            }
        }

        private static void Draw3DLineInViewport(Vector3 a, Vector3 b, Vector3 camPos, Quaternion camRot, float focal, Rect viewport, Color color, float thickness)
        {
            if (TryProject3DPoint(a, camPos, camRot, focal, viewport, out Vector2 sa, out _) &&
                TryProject3DPoint(b, camPos, camRot, focal, viewport, out Vector2 sb, out _))
            {
                Draw2DLineClipped(sa, sb, color, thickness);
            }
        }

        private static void Draw2DLineClipped(Vector2 a, Vector2 b, Color color, float thickness)
        {
            Vector2 delta = b - a;
            float len = delta.magnitude;
            if (len < 0.5f)
            {
                return;
            }

            Matrix4x4 prevMatrix = GUI.matrix;
            Color prevColor = GUI.color;

            GUI.color = color;
            float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
            GUIUtility.RotateAroundPivot(angle, a);
            GUI.DrawTexture(new Rect(a.x, a.y - thickness * 0.5f, len, thickness), viewerWhiteTex);

            GUI.matrix = prevMatrix;
            GUI.color = prevColor;
        }
    }
}
