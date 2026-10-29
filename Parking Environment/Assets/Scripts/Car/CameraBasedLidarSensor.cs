using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityStandardAssets.Vehicles.Car
{
    /// <summary>
    /// Camera-Based 3D LiDAR sensor for the ego car.
    /// Emulates a multi-channel 3D LiDAR using standard Unity Cameras and depth/range rendering
    /// (without requiring DXR/HDRP raytracing hardware), following the Unity SensorSDK CameraBasedLidar
    /// and Unity Simulation LiDAR configuration specifications.
    /// </summary>
    public class CameraBasedLidarSensor : MonoBehaviour
    {
        public enum LidarSamplingMode
        {
            /// <summary>
            /// Samples the scene using Unity Cameras and the Hidden/CameraBasedLidarDepth replacement shader
            /// (automatically falls back to PhysicsRaycast when running in headless -nographics mode).
            /// </summary>
            CameraDepthShader = 0,

            /// <summary>
            /// Samples the scene using physics raycasts along each camera's pixel frustum rays.
            /// </summary>
            PhysicsRaycast = 1
        }

        public enum LidarOutputType
        {
            /// <summary>
            /// Outputs 1 float per laser beam: radial distance normalized to [0, 1] by maxRange (1.0 = no hit).
            /// </summary>
            LaserRangeScan = 0,

            /// <summary>
            /// Outputs 3 floats per laser beam: ego-car-local (x, y, z) hit coordinates normalized by maxRange.
            /// </summary>
            PointCloudXYZ = 1
        }

        [Header("LiDAR Camera Rig")]
        [Tooltip("Cameras forming the LiDAR rig (defaults to the 8 surround cameras under Self_driving_cameras).")]
        public Camera[] lidarCameras;

        [Header("Scan Pattern & Geometry")]
        [Tooltip("Number of horizontal laser samples per camera (e.g., 16 samples x 8 cameras = 128 horizontal samples).")]
        [Range(1, 256)]
        public int horizontalSamplesPerCamera = 16;

        [Tooltip("Number of vertical laser channels (layers) per camera.")]
        [Range(1, 64)]
        public int verticalChannels = 4;

        [Tooltip("Horizontal field of view in degrees per camera.")]
        [Range(10f, 160f)]
        public float horizontalFov = 90f;

        [Tooltip("Minimum vertical elevation angle in degrees.")]
        [Range(-60f, 0f)]
        public float minVerticalAngle = -2f;

        [Tooltip("Maximum vertical elevation angle in degrees.")]
        [Range(0f, 60f)]
        public float maxVerticalAngle = 10f;

        [Header("Ground & Trigger Filtering")]
        [Tooltip("Exclude flat ground Plane and non-solid trigger objects (such as the green Target parking zone) from LiDAR obstacle returns.")]
        public bool ignoreGroundAndTriggers = true;

        [Tooltip("Height threshold (in meters relative to ego car base) below which hits are filtered as road surface.")]
        [Range(-0.2f, 0.5f)]
        public float groundHeightThreshold = 0.15f;

        [Header("Range & Timing")]
        [Tooltip("Minimum detection range in meters (near clip plane).")]
        public float minRange = 0.3f;

        [Tooltip("Maximum detection range in meters (far clip plane).")]
        public float maxRange = 20f;

        [Tooltip("LiDAR scan update rate in Hz (revolutions/scans per second). Standard automotive LiDAR is 10 Hz.")]
        public float rotationFrequencyHz = 10f;

        [Header("Output & Noise Configuration")]
        [Tooltip("Sampling implementation: GPU Camera Depth Shader (Raster LiDAR) or Physics Raycast.")]
        public LidarSamplingMode samplingMode = LidarSamplingMode.CameraDepthShader;

        [Tooltip("Output format sent to the ML-Agents VectorSensor.")]
        public LidarOutputType outputType = LidarOutputType.LaserRangeScan;

        [Tooltip("Standard deviation of Gaussian range noise in meters (0 = noise-free).")]
        [Range(0f, 0.5f)]
        public float rangeNoiseStdDev = 0f;

        [Tooltip("Layer mask of objects detectable by the LiDAR.")]
        public LayerMask cullingMask = ~0;

        [Header("Debug & Point Cloud Visualization")]
        [Tooltip("Draw 3D LiDAR hit points (point cloud) in the Scene View (and Game View if Gizmos are enabled).")]
        public bool drawPointCloudGizmos = true;

        [Tooltip("Draw the laser beam lines from the cameras to the hit points in the Scene View.")]
        public bool drawDebugRays = false;

        [Tooltip("Radius of each rendered 3D point cloud sphere in meters.")]
        [Range(0.02f, 0.3f)]
        public float pointCloudSphereRadius = 0.08f;

        [Tooltip("Show a live Bird's-Eye-View (BEV) point cloud radar and 360° depth map HUD in the Game View.")]
        public bool showPointCloudHud = true;

        private Shader lidarDepthShader;
        private RenderTexture scanRenderTexture;
        private Texture2D readbackTexture;
        private Texture2D hudPreviewTexture;
        private Color[] hudPreviewPixels;
        private bool useRFloatTexture;

        private float[] normalizedRanges;
        private float[] observationBuffer;
        private Vector3[] lastWorldRayOrigins;
        private Vector3[] lastWorldRayDirections;
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

        /// <summary>
        /// Number of active cameras in the LiDAR rig.
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
        /// Total number of laser beams across all cameras (CameraCount * horizontalSamplesPerCamera * verticalChannels).
        /// </summary>
        public int TotalBeamCount => Mathf.Max(1, CameraCount) * Mathf.Max(1, horizontalSamplesPerCamera) * Mathf.Max(1, verticalChannels);

        /// <summary>
        /// Total number of floats written to the ML-Agents VectorSensor per step.
        /// </summary>
        public int ObservationSize => outputType == LidarOutputType.PointCloudXYZ ? TotalBeamCount * 3 : TotalBeamCount;

        /// <summary>
        /// Read-only access to the latest normalized range values [0, 1] for each beam.
        /// </summary>
        public float[] LatestNormalizedRanges => normalizedRanges;

        /// <summary>
        /// Initializes the LiDAR camera rig, render targets, and buffers.
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

            // Load depth replacement shader
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
        /// Performs or retrieves the current LiDAR scan and returns the flattened 1D float array for ML-Agents VectorSensor.
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
            verticalChannels = Mathf.Max(1, verticalChannels);
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
                // Each camera renders into its own horizontal slice of the shared atlas RenderTexture
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
            int height = Mathf.Max(1, verticalChannels);
            int totalBeams = TotalBeamCount;
            int obsSize = ObservationSize;

            normalizedRanges = new float[totalBeams];
            observationBuffer = new float[obsSize];
            lastWorldRayOrigins = new Vector3[totalBeams];
            lastWorldRayDirections = new Vector3[totalBeams];
            lastHitDistances = new float[totalBeams];

            for (int i = 0; i < totalBeams; i++)
            {
                normalizedRanges[i] = 1f;
                lastHitDistances[i] = maxRange;
            }

            ReleaseTextures();

            useRFloatTexture = SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat);
            RenderTextureFormat rtFormat = useRFloatTexture ? RenderTextureFormat.RFloat : RenderTextureFormat.ARGB32;
            TextureFormat texFormat = useRFloatTexture ? TextureFormat.RFloat : TextureFormat.RGBA32;

            scanRenderTexture = new RenderTexture(atlasWidth, height, 16, rtFormat, RenderTextureReadWrite.Linear)
            {
                name = "LidarScanAtlasRT",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                antiAliasing = 1
            };
            scanRenderTexture.Create();

            readbackTexture = new Texture2D(atlasWidth, height, texFormat, false, true)
            {
                name = "LidarAtlasReadbackTex",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            hudPreviewTexture = new Texture2D(atlasWidth, height, TextureFormat.RGBA32, false, false)
            {
                name = "LidarHudPreviewTex",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            hudPreviewPixels = new Color[atlasWidth * height];
        }

        private void CaptureScan()
        {
            if (lidarCameras == null || lidarCameras.Length == 0)
            {
                return;
            }

            bool canUseGpuShader = samplingMode == LidarSamplingMode.CameraDepthShader &&
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
        }

        private void CaptureScanWithCameraShader()
        {
            int camCount = lidarCameras.Length;
            int singleWidth = Mathf.Max(1, horizontalSamplesPerCamera);
            int atlasWidth = camCount * singleWidth;
            int height = Mathf.Max(1, verticalChannels);

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

            // Single GPU-to-CPU ReadPixels call for the entire 8-camera atlas
            RenderTexture.active = scanRenderTexture;
            readbackTexture.ReadPixels(new Rect(0, 0, atlasWidth, height), 0, 0, false);
            RenderTexture.active = previousActive;

            Color[] pixels = readbackTexture.GetPixels();
            bool updateHudPreview = showPointCloudHud && primaryHudSensor == this && hudPreviewTexture != null && hudPreviewPixels != null;

            int beamIndex = 0;
            int obsIndex = 0;

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
                    float viewportV = height == 1 ? 0.5f : (v + 0.5f) / height;
                    int rowOffset = v * atlasWidth + xOffset;

                    for (int u = 0; u < singleWidth; u++)
                    {
                        float viewportU = singleWidth == 1 ? 0.5f : (u + 0.5f) / singleWidth;
                        int pixelIdx = rowOffset + u;
                        Color pixel = pixels[pixelIdx];

                        float normDist = Mathf.Clamp01(pixel.r);
                        float distanceMeters = normDist * maxRange;

                        if (rangeNoiseStdDev > 0f && normDist < 0.999f)
                        {
                            distanceMeters = Mathf.Clamp(distanceMeters + SampleGaussianNoise(rangeNoiseStdDev), minRange, maxRange);
                            normDist = distanceMeters / maxRange;
                        }

                        bool needWorldRay = ignoreGroundAndTriggers || drawPointCloudGizmos || drawDebugRays || showPointCloudHud || outputType == LidarOutputType.PointCloudXYZ;
                        if (needWorldRay)
                        {
                            Ray worldRay = GetBeamWorldRay(cam, viewportU, viewportV);

                            if (ignoreGroundAndTriggers && normDist < 0.999f)
                            {
                                Vector3 worldHit = worldRay.origin + worldRay.direction * distanceMeters;
                                Vector3 localHit = transform.InverseTransformPoint(worldHit);
                                if (localHit.y <= groundHeightThreshold)
                                {
                                    normDist = 1f;
                                    distanceMeters = maxRange;
                                }
                            }

                            lastWorldRayOrigins[beamIndex] = worldRay.origin;
                            lastWorldRayDirections[beamIndex] = worldRay.direction;
                            lastHitDistances[beamIndex] = distanceMeters;
                            WriteObservationValue(worldRay, normDist, distanceMeters, ref obsIndex);
                        }
                        else
                        {
                            observationBuffer[obsIndex++] = normDist;
                        }

                        if (updateHudPreview && pixelIdx < hudPreviewPixels.Length)
                        {
                            hudPreviewPixels[pixelIdx] = normDist < 0.999f
                                ? Color.HSVToRGB(Mathf.Clamp01(normDist * 0.35f), 0.95f, 1f)
                                : new Color(0.05f, 0.07f, 0.1f, 1f);
                        }

                        normalizedRanges[beamIndex] = normDist;
                        beamIndex++;
                    }
                }
            }

            if (updateHudPreview)
            {
                hudPreviewTexture.SetPixels(hudPreviewPixels);
                hudPreviewTexture.Apply(false);
            }
        }

        private void CaptureScanWithPhysicsRaycast()
        {
            int width = Mathf.Max(1, horizontalSamplesPerCamera);
            int height = Mathf.Max(1, verticalChannels);

            int beamIndex = 0;
            int obsIndex = 0;

            SetEgoCollidersEnabled(false);
            try
            {
                for (int camIdx = 0; camIdx < lidarCameras.Length; camIdx++)
                {
                    Camera cam = lidarCameras[camIdx];
                    if (cam == null)
                    {
                        continue;
                    }

                    for (int v = 0; v < height; v++)
                    {
                        float viewportV = height == 1 ? 0.5f : (v + 0.5f) / height;
                        for (int u = 0; u < width; u++)
                        {
                            float viewportU = width == 1 ? 0.5f : (u + 0.5f) / width;
                            Ray worldRay = GetBeamWorldRay(cam, viewportU, viewportV);

                            float distanceMeters = maxRange;
                            float normDist = 1f;

                            if (Physics.Raycast(worldRay.origin, worldRay.direction, out RaycastHit hit, maxRange, cullingMask, QueryTriggerInteraction.Ignore))
                            {
                                bool isGroundHit = false;
                                if (ignoreGroundAndTriggers)
                                {
                                    Vector3 localHit = transform.InverseTransformPoint(hit.point);
                                    if (localHit.y <= groundHeightThreshold ||
                                        hit.collider.gameObject.name == "Plane" ||
                                        hit.collider.CompareTag("Finish"))
                                    {
                                        isGroundHit = true;
                                    }
                                }

                                if (!isGroundHit)
                                {
                                    distanceMeters = Mathf.Clamp(hit.distance, minRange, maxRange);
                                    if (rangeNoiseStdDev > 0f)
                                    {
                                        distanceMeters = Mathf.Clamp(distanceMeters + SampleGaussianNoise(rangeNoiseStdDev), minRange, maxRange);
                                    }
                                    normDist = distanceMeters / maxRange;
                                }
                            }

                            normalizedRanges[beamIndex] = normDist;
                            lastWorldRayOrigins[beamIndex] = worldRay.origin;
                            lastWorldRayDirections[beamIndex] = worldRay.direction;
                            lastHitDistances[beamIndex] = distanceMeters;

                            WriteObservationValue(worldRay, normDist, distanceMeters, ref obsIndex);
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

        private void WriteObservationValue(Ray worldRay, float normDist, float distanceMeters, ref int obsIndex)
        {
            if (outputType == LidarOutputType.PointCloudXYZ)
            {
                if (normDist >= 0.999f)
                {
                    observationBuffer[obsIndex++] = 0f;
                    observationBuffer[obsIndex++] = 0f;
                    observationBuffer[obsIndex++] = 0f;
                }
                else
                {
                    Vector3 worldHitPoint = worldRay.origin + worldRay.direction * distanceMeters;
                    Vector3 localHitPoint = transform.InverseTransformPoint(worldHitPoint) / maxRange;
                    observationBuffer[obsIndex++] = Mathf.Clamp(localHitPoint.x, -1f, 1f);
                    observationBuffer[obsIndex++] = Mathf.Clamp(localHitPoint.y, -1f, 1f);
                    observationBuffer[obsIndex++] = Mathf.Clamp(localHitPoint.z, -1f, 1f);
                }
            }
            else
            {
                observationBuffer[obsIndex++] = normDist;
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
            // Always hide ego car's own body meshes from its onboard LiDAR cameras
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

            if (hudPreviewTexture != null)
            {
                Destroy(hudPreviewTexture);
                hudPreviewTexture = null;
            }
        }

        private static CameraBasedLidarSensor primaryHudSensor;
        private static Texture2D hudWhiteTex;

        private void OnEnable()
        {
            if (primaryHudSensor == null)
            {
                primaryHudSensor = this;
            }
        }

        private void OnDisable()
        {
            if (primaryHudSensor == this)
            {
                primaryHudSensor = null;
            }
        }

        private void LateUpdate()
        {
            // Ensure live visualization updates even during heuristic spot-finding phases before RequestDecision() starts
            if (isInitialized && primaryHudSensor == this && (showPointCloudHud || drawPointCloudGizmos || drawDebugRays))
            {
                float interval = rotationFrequencyHz > 0f ? (1f / rotationFrequencyHz) : 0.1f;
                if (Time.time - lastScanTime >= interval)
                {
                    Perceive();
                }
            }
        }

        private void OnDestroy()
        {
            if (primaryHudSensor == this)
            {
                primaryHudSensor = null;
            }
            ReleaseTextures();
        }

        private void OnDrawGizmos()
        {
            if ((!drawPointCloudGizmos && !drawDebugRays) ||
                lastWorldRayOrigins == null || lastWorldRayDirections == null || normalizedRanges == null)
            {
                return;
            }

            // In multi-agent scenes, only draw gizmos for the primary agent or when selected
            if (primaryHudSensor != null && primaryHudSensor != this)
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
                    Color hitColor = Color.HSVToRGB(Mathf.Clamp01(norm * 0.35f), 0.95f, 1f); // Red (close) -> Yellow -> Green (far)
                    Gizmos.color = hitColor;

                    if (drawPointCloudGizmos)
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
            if (!showPointCloudHud || primaryHudSensor != this || normalizedRanges == null || lastWorldRayOrigins == null)
            {
                return;
            }

            if (hudWhiteTex == null)
            {
                hudWhiteTex = new Texture2D(1, 1);
                hudWhiteTex.SetPixel(0, 0, Color.white);
                hudWhiteTex.Apply();
            }

            const float margin = 12f;
            const float radarSize = 220f;
            const float depthStripHeight = 44f;
            Rect panelRect = new Rect(margin, margin, radarSize + 16f, radarSize + depthStripHeight + 52f);

            // Background panel
            GUI.color = new Color(0.08f, 0.09f, 0.12f, 0.85f);
            GUI.DrawTexture(panelRect, hudWhiteTex);
            GUI.color = Color.white;

            GUIStyle titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            GUI.Label(new Rect(panelRect.x, panelRect.y + 4f, panelRect.width, 18f),
                $"LiDAR Point Cloud ({TotalBeamCount} beams)", titleStyle);

            // Bird's-Eye-View (BEV) Radar Box
            Rect radarRect = new Rect(panelRect.x + 8f, panelRect.y + 24f, radarSize, radarSize);
            GUI.color = new Color(0.03f, 0.04f, 0.06f, 0.95f);
            GUI.DrawTexture(radarRect, hudWhiteTex);

            Vector2 center = radarRect.center;
            float halfRadar = radarSize * 0.5f;

            // Draw ego car icon at center of radar
            float carWidthPx = Mathf.Max(6f, (1.8f / maxRange) * halfRadar);
            float carLengthPx = Mathf.Max(12f, (4.2f / maxRange) * halfRadar);
            GUI.color = new Color(0.2f, 0.6f, 1f, 0.9f);
            GUI.DrawTexture(new Rect(center.x - carWidthPx * 0.5f, center.y - carLengthPx * 0.5f, carWidthPx, carLengthPx), hudWhiteTex);

            // Plot live 3D hit points projected into ego-car top-down (X, Z) coordinates
            int count = Mathf.Min(lastWorldRayOrigins.Length, normalizedRanges.Length);
            for (int i = 0; i < count; i++)
            {
                float norm = normalizedRanges[i];
                if (norm >= 0.999f)
                {
                    continue;
                }

                float dist = lastHitDistances != null && i < lastHitDistances.Length ? lastHitDistances[i] : norm * maxRange;
                Vector3 worldHit = lastWorldRayOrigins[i] + lastWorldRayDirections[i] * dist;
                Vector3 localHit = transform.InverseTransformPoint(worldHit);

                float px = center.x + (localHit.x / maxRange) * halfRadar;
                float py = center.y - (localHit.z / maxRange) * halfRadar;

                if (radarRect.Contains(new Vector2(px, py)))
                {
                    GUI.color = Color.HSVToRGB(Mathf.Clamp01(norm * 0.35f), 0.95f, 1f);
                    GUI.DrawTexture(new Rect(px - 2f, py - 2f, 4f, 4f), hudWhiteTex);
                }
            }

            // Draw 360° Panoramic Depth Atlas below the BEV radar
            Texture2D stripTex = hudPreviewTexture != null ? hudPreviewTexture : readbackTexture;
            if (stripTex != null)
            {
                Rect stripLabelRect = new Rect(panelRect.x + 8f, radarRect.yMax + 2f, radarSize, 16f);
                GUI.color = Color.white;
                GUI.Label(stripLabelRect, "360° Depth Map (8 Cams × 4 Layers):");

                Rect stripRect = new Rect(panelRect.x + 8f, radarRect.yMax + 18f, radarSize, depthStripHeight);
                GUI.DrawTexture(stripRect, stripTex, ScaleMode.StretchToFill, false);
            }

            GUI.color = Color.white;
        }
    }
}
