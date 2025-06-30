"use client";

import { ToggleButton, Tooltip } from "@mui/material";
import React from "react";

interface SimpleToggleButtonProps {
    selected?: boolean;
    onChange?: (event: React.MouseEvent<HTMLElement>, value: unknown) => void;
    icon?: React.ReactElement;
    tooltip?: string;
}

export default function SimpleToggleButton({selected, onChange, icon, tooltip}: SimpleToggleButtonProps) {
    return (
        <Tooltip
            title={tooltip || ""}
            arrow>
            <ToggleButton
                value="square"
                selected={selected}
                onChange={onChange}
            >
                {icon}
            </ToggleButton>

        </Tooltip>

    )
}