Shader "Hidden/CameraBasedLidarDepth"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            ZWrite On
            ZTest LEqual
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float _LidarMaxRange;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 viewPos : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                // View-space coordinates from the camera optical center
                o.viewPos = UnityObjectToViewPos(v.vertex);
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // True 3D Euclidean radial distance from camera origin to surface hit point
                float radialDistance = length(i.viewPos);
                float maxRange = max(_LidarMaxRange, 0.0001);
                float normalizedRange = saturate(radialDistance / maxRange);
                return float4(normalizedRange, normalizedRange, normalizedRange, 1.0);
            }
            ENDCG
        }
    }
    FallBack Off
}

