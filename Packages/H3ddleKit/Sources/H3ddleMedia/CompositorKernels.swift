import CoreImage
import Foundation

/// Metal Core Image kernels for effects that stock CI filters cannot express.
/// Compiled at first use via `CIKernel` — not the deprecated CIKL source path.
enum CompositorKernels {
  // Compiled kernels are immutable, Sendable values under the macOS 26 SDK.
  // These lets are initialized once and only read afterwards.
  static let grain: CIKernel? =
    loadResult.kernels.first { $0.name == "h3FilmGrain" }
  static let chroma: CIColorKernel? =
    loadResult.kernels.compactMap { $0 as? CIColorKernel }
    .first { $0.name == "h3ChromaKey" }
  static var compileError: String? { loadResult.error }

  private static let loadResult:
    (kernels: [CIKernel], error: String?) = {
      do {
        return (try CIKernel.kernels(withMetalString: metalSource), nil)
      } catch {
        return ([], String(describing: error))
      }
    }()

  private static let metalSource = """
    #include <CoreImage/CoreImage.h>
    using namespace metal;

    [[ stitchable ]] half4 h3FilmGrain(
        coreimage::sample_h s,
        float amount,
        float cells,
        float frame,
        float width,
        float height,
        coreimage::destination dest
    ) {
        float2 dc = dest.coord();
        float2 size = max(float2(width, height), float2(1.0));
        float2 uv = dc / size;
        float2 cell = floor(uv * max(8.0, cells)) + float2(frame, frame);
        float n = fract(sin(dot(cell, float2(12.9898, 78.233))) * 43758.5453) - 0.5;
        float luma = dot(float3(s.rgb), float3(0.2126, 0.7152, 0.0722));
        float mid = luma * (1.0 - luma) * 4.0;
        float3 rgb = float3(s.rgb) + n * amount * mid;
        return half4(half3(clamp(rgb, 0.0, 1.0)), s.a);
    }

    [[ stitchable ]] half4 h3ChromaKey(coreimage::sample_h s, float targetBlue, float softness) {
        float3 rgb = float3(s.rgb);
        float primary = targetBlue > 0.5 ? rgb.b : rgb.g;
        float others = targetBlue > 0.5 ? max(rgb.r, rgb.g) : max(rgb.r, rgb.b);
        float dominance = primary - others;
        float sat = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
        float edge = 0.04 + softness * 0.35;
        float mask = 1.0 - smoothstep(0.02, edge, dominance)
            * smoothstep(0.04, 0.18 + softness * 0.25, sat);
        return half4(s.rgb * half(mask), s.a * half(mask));
    }
    """
}
