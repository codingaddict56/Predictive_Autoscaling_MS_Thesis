import { vmcloudMapper } from "./vmcloud.js";
import { borgMapper } from "./borg.js";
import { cicidsMapper } from "./cicids.js";

export function createMapper(kind) {
  switch (kind) {
    case "vmcloud": return vmcloudMapper;
    case "borg": return borgMapper;
    case "cicids": return cicidsMapper;
    default: return vmcloudMapper;
  }
}
